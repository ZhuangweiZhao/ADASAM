"""
iSAID-5i 小样本语义分割训练 | Few-shot Semantic Segmentation Training.
======================================================================

基于 AdaSAM Stage 2 架构 (DPG + SupportEncoder + SAM Decoder) 在 iSAID-5i
标准小样本协议上训练。

Train AdaSAM Stage 2 architecture (DPG + SupportEncoder + SAM Decoder) on
the standard iSAID-5i few-shot protocol (15 classes, 3-fold cross-validation).

用法 | Usage::

    python tools/train_isaid_5i.py --fold 0 --k-shot 5 --epochs 50         # 5-shot fold 0
    python tools/train_isaid_5i.py --fold 1 --k-shot 1 --epochs 50         # 1-shot fold 1
    python tools/train_isaid_5i.py --fold 0 --k-shot 5 --epochs 1 --steps 5  # smoke test
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# 将 repo 根目录加入 sys.path, 确保可直接导入 adasam 包
# Add repo root to sys.path so adasam imports work without pip install -e
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adasam.adapters import CATAdapter
from adasam.backbone import build_mobile_sam, MobileSAMBackbone
from adasam.datasets.isaid_5i import (
    ISAID5iDataset,
    ISAID5iEpisodeSampler,
    ISAID5I_CATEGORIES,
    ISAID5I_FOLDS,
)
from adasam.logging import get_logger
from adasam.logging.backends import ConsoleBackend, FileBackend
from adasam.losses import CriterionConfig, HungarianMatcher, MatcherConfig, SetCriterion
from adasam.model import AdaSAMModel, AdaSAMModelConfig
from adasam.utils import set_seed
from adasam.utils.transforms import preprocess_image, resize_mask


# ═══════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════

class ISAID5iTrainer:
    """iSAID-5i 小样本训练器 | Few-shot trainer for iSAID-5i."""

    def __init__(self, cfg: dict, args: argparse.Namespace) -> None:
        """初始化训练器 | Initialize the trainer.

        完成以下设置: 随机种子/设备、数据集+Episode Sampler、模型+Backbone、
        Loss+Matcher、优化器+学习率调度器、输出目录+日志。
        Sets up: random seed/device, dataset + episode sampler, model + backbone,
        loss + matcher, optimizer + scheduler, output dir + logger.
        """
        self.cfg = cfg
        self.args = args
        self.seed = int(cfg.get("seed", 42))
        set_seed(self.seed)
        self.device = torch.device(
            cfg["train"].get("device", "cuda") if torch.cuda.is_available() else "cpu"
        )
        self._rng = random.Random(self.seed)

        # ── 数据 | Data ──
        self.fold = int(cfg["data"].get("fold", 0))
        self.k_shot = int(cfg["fewshot"].get("k_shot", 5))
        self.mode = cfg["fewshot"].get("train_mode", "novel")
        data_root = self._resolve(cfg["data"]["data_root"])

        # 训练数据集 + 类别统计 | Training dataset + class stats
        self.train_ds = ISAID5iDataset(root=data_root, fold=self.fold, split="train", mode=self.mode)
        self.train_classes = self.train_ds.visible_classes()
        print(f"[iSAID-5i] fold={self.fold} mode={self.mode} classes={self.train_classes}")
        for cls in self.train_classes:
            n = len(self.train_ds.class_to_tiles(cls))
            name = ISAID5I_CATEGORIES.get(cls, f"cls{cls}")
            print(f"  class {cls:>2d} ({name:<20s}): {n} tiles")

        # Episode Sampler — 保证 support 和 query 来自不同场景 | ensures scene-disjoint
        self.sampler = ISAID5iEpisodeSampler(
            self.train_ds, k_shot=self.k_shot, seed=self.seed,
            min_tiles=int(cfg["fewshot"].get("min_tiles", 10)),
        )
        self.eligible = self.sampler.eligible_classes()
        print(f"[iSAID-5i] eligible classes after filtering: {len(self.eligible)}")

        # ── 验证集 | Validation ──
        tcfg = cfg["train"]
        self.val_every = int(tcfg.get("val_every", 10))
        self.val_samples = int(tcfg.get("val_samples", 30))
        self.val_ds = ISAID5iDataset(root=data_root, fold=self.fold, split="val", mode=self.mode)
        self.val_classes = self.val_ds.visible_classes()
        print(f"[iSAID-5i] val classes: {len(self.val_classes)}, "
              f"val_every={self.val_every}, val_samples={self.val_samples}")

        # ── 模型 | Model ──
        # MobileSAM backbone 始终冻结 | MobileSAM backbone is always frozen
        ckpt_path = self._resolve(cfg["backbone"]["checkpoint"])
        sam = build_mobile_sam(ckpt_path, cfg["backbone"].get("model_type", "vit_t"), self.device)
        self.backbone = MobileSAMBackbone(sam.image_encoder, sam.image_encoder.img_size).to(self.device)
        self.image_size = self.backbone.img_size
        self.embed_dim = int(cfg.get("support_encoder", {}).get("embed_dim", 256))
        self.model = AdaSAMModel(sam, AdaSAMModelConfig.from_dict(cfg)).to(self.device)
        self.num_queries = self.model.num_queries

        # ── 损失函数 | Criterion ──
        # Hungarian Matcher + Focal/Dice/Objectness loss
        loss_cfg = cfg.get("loss", {})
        self.criterion = SetCriterion(
            HungarianMatcher(MatcherConfig.from_dict(loss_cfg)),
            CriterionConfig.from_dict(loss_cfg),
        )

        # ── 优化器 & 学习率调度 | Optimizer & LR Scheduler ──
        tcfg = cfg["train"]
        self.epochs = int(tcfg.get("epochs", 50))
        self.episodes_per_epoch = int(tcfg.get("episodes_per_epoch", 200))
        self.grad_clip = float(tcfg.get("grad_clip", 1.0))
        lr = float(tcfg.get("lr", 1e-4))
        sam_mult = float(tcfg.get("sam_decoder_lr_mult", 0.1))

        # CAT-Adapter (可选 | optional): 轻量特征适配 | lightweight feature adaptation
        self.cat_adapter = None
        if bool(tcfg.get("use_cat_adapter", False)):
            adapter_cfg = tcfg.get("cat_adapter", {})
            self.cat_adapter = CATAdapter(
                dim=self.embed_dim,
                bottleneck=int(adapter_cfg.get("bottleneck", 64)),
            ).to(self.device)

        # 参数分组: DPG + SupportEncoder 全速, SAM Decoder 低速 (×sam_mult)
        # Param groups: DPG + SupportEncoder full-rate, SAM Decoder reduced (×sam_mult)
        param_groups = [
            {"params": list(self.model.dpg.parameters()), "lr": lr},
            {"params": list(self.model.support_encoder.parameters()), "lr": lr},
            {"params": [
                p for p in self.model.sam_decoder.mask_decoder.parameters()
                if p.requires_grad
            ], "lr": lr * sam_mult},
        ]
        if self.model.coarse_prior is not None:
            param_groups.append(
                {"params": list(self.model.coarse_prior.parameters()), "lr": lr}
            )
        if self.cat_adapter is not None:
            param_groups.append({"params": list(self.cat_adapter.parameters()), "lr": lr})
        # 收集所有可训练参数 (用于 gradient clipping)
        # Collect all trainable params (for gradient clipping)
        self._trainable = [p for g in param_groups for p in g["params"]]
        self.optimizer = AdamW(
            param_groups, lr=lr, weight_decay=float(tcfg.get("weight_decay", 1e-4))
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        # ── 输出目录 & 日志 | Output & Logging ──
        exp = f"isaid5i_fold{self.fold}_k{self.k_shot}_{self.mode}_seed{self.seed}"
        self.out_dir = self._resolve(cfg.get("output_dir", "runs")) / exp
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("trainer.isaid5i")
        if not self.logger.backends:
            self.logger.add_backend(ConsoleBackend())
            self.logger.add_backend(FileBackend(str(self.out_dir / "train.jsonl")))

        n_train = sum(p.numel() for p in self._trainable) / 1e6
        self.logger.log_info(
            "init",
            f"fold={self.fold} k={self.k_shot} mode={self.mode} device={self.device} "
            f"trainable={n_train:.2f}M queries={self.num_queries} "
            f"classes={self.eligible} out={self.out_dir}",
        )

    @staticmethod
    def _resolve(path: str | Path) -> Path:
        """将相对路径转为相对于 repo 根目录的绝对路径 | Resolve relative path to repo root."""
        p = Path(path)
        return p if p.is_absolute() else (_REPO_ROOT / p)

    # ── 特征提取 | Embedding ──

    def _embed(self, image: torch.Tensor) -> torch.Tensor:
        """图像 → 特征图 | Image → feature map.

        预处理 (resize+normalize) → MobileSAM backbone (frozen) → 可选 CAT-Adapter.
        Preprocess (resize+normalize) → MobileSAM backbone (frozen) → optional CAT-Adapter.
        """
        x, _ = preprocess_image(image)
        emb = self.backbone(x.unsqueeze(0).to(self.device))["image_embedding"]
        if self.cat_adapter is not None:
            emb = self.cat_adapter(emb)
        return emb

    # ── Support Memory 构建 | Build Support Memory ──

    def _build_support_memory(
        self, support_indices: list[int], class_id: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """从 K 张 support tile 构建 (features, masks) 对 | Build (features, masks) from K support tiles.

        :return: (support_features [K,C,64,64], support_masks [K,64,64]) 或 None (无效样本).
        """
        images, masks = [], []
        for idx in support_indices:
            sample = self.train_ds[idx]
            fg = self._class_foreground(sample["instances"], class_id)
            if fg is None:
                continue
            x, _ = preprocess_image(sample["image"])
            images.append(x.to(self.device))
            masks.append(fg)
        if not images:
            return None  # 无有效 support 图像 | no valid support images

        # 批量提取 backbone 特征 + 将 mask resize 到 64² 网格
        # Batch backbone features + resize masks to 64² grid
        feats = self.backbone(torch.stack(images, dim=0))["image_embedding"]  # [K,256,64,64]
        if self.cat_adapter is not None:
            feats = self.cat_adapter(feats)
        masks_grid = torch.stack(
            [resize_mask(m, (feats.shape[2], feats.shape[3])).to(self.device) for m in masks],
            dim=0,
        )
        if masks_grid.sum() < 1.0:
            return None  # support 前景为空 | empty foreground
        return feats, masks_grid

    @staticmethod
    def _class_foreground(instances: list[dict], class_id: int) -> torch.Tensor | None:
        """合并指定类别的所有实例 mask 为单张前景图 | Merge all instance masks of a class into one FG map.

        :return: [H,W] float tensor 或 None (该类别不存在于该 tile).
        """
        fg = None
        for inst in instances:
            if inst["category_id"] == class_id:
                if fg is None:
                    fg = inst["mask"].clone()
                else:
                    fg = fg | inst["mask"]
        return fg.float() if fg is not None else None

    # ── 单 Episode 训练 | Single Episode Training ──

    def _train_episode(self, episode: dict) -> dict | None:
        """执行一个 episode 的前向+反向 | Run one episode: forward + backward.

        流程: support memory → query embedding → DPG → SAM Decoder → loss → backward.
        Flow: support memory → query embedding → DPG → SAM Decoder → loss → backward.

        :return: loss 指标字典, 或 None (episode 无效 | invalid episode).
        """
        cls = episode["class_id"]

        # 构建 support 特征 (K 张 support tile) | Build support features (K support tiles)
        support_data = self._build_support_memory(episode["support_indices"], cls)
        if support_data is None:
            return None
        support_features, support_masks_grid = support_data

        # 提取 query GT mask (仅当前类别) | Extract query GT masks (current class only)
        query = self.train_ds[episode["query_index"]]
        gt_list = [i["mask"] for i in query["instances"] if i["category_id"] == cls]
        if not gt_list:
            return None
        if len(gt_list) > self.num_queries:
            # 按面积保留 top-n (大实例优先), 小碎片为噪声 | Keep top-n by area, small fragments = noise
            gt_list = sorted(gt_list, key=lambda m: m.sum(), reverse=True)[:self.num_queries]
        gt_masks = torch.stack([m.float() for m in gt_list], dim=0).to(self.device)

        # 前向传播 + 损失计算 | Forward + loss
        emb = self._embed(query["image"])
        dpg_out, low_res, iou_pred = self.model.forward_train(
            emb, support_features, support_masks_grid
        )
        losses = self.criterion(low_res[:, 0], iou_pred[:, 0], dpg_out, gt_masks)

        # 反向传播 | Backward
        self.optimizer.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self._trainable, self.grad_clip)
        self.optimizer.step()

        return {
            "loss": float(losses["loss"].detach()),
            "focal": float(losses["focal"]),
            "dice": float(losses["dice"]),
            "obj": float(losses["obj"]),
            "iou_head": float(losses["iou_head"]),
            "aux": float(losses["aux"]),
            "prompt_focal": float(losses["prompt_focal"]),
            "prompt_dice": float(losses["prompt_dice"]),
            "prompt": float(losses["prompt"]),
            "n_matched": float(losses["n_matched"]),
            "mean_obj_matched": float(losses["mean_obj_matched"]),
            "mean_obj_unmatched": float(losses["mean_obj_unmatched"]),
            "n_inst": gt_masks.shape[0],
        }

    # ── Validation ──

    @torch.no_grad()
    def _build_val_support_cache(
        self
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """为每个验证类别构建固定的 support (features, masks) — FSS 标准协议.
        Build fixed support (features, masks) per class for validation (FSS standard).

        :return: {class_id: (support_features [K,C,gh,gw], support_masks [K,gh,gw])}
        """
        cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        val_rng = random.Random(self.seed + 1000)

        for cls in self.val_classes:
            tiles = self.train_ds.class_to_tiles(cls)
            if len(tiles) < 1:
                continue

            # Sample K support tiles (scene-disjoint)
            scenes: dict[str, list[int]] = defaultdict(list)
            for idx in tiles:
                tile_id = self.train_ds.tile_ids[idx]
                src = self.train_ds._source_images.get(tile_id, str(idx))
                scenes[src].append(idx)

            chosen = []
            scene_list = list(scenes.keys())
            val_rng.shuffle(scene_list)
            for src in scene_list:
                if len(chosen) >= self.k_shot:
                    break
                idx = val_rng.choice(scenes[src])
                chosen.append(idx)

            if len(chosen) < 1:
                continue

            # Build support features + masks
            support_data = self._build_support_memory(chosen, cls)
            if support_data is None:
                continue
            sup_feat, sup_mask = support_data  # [K,C,64,64], [K,64,64]
            cache[cls] = (sup_feat, sup_mask)

        n_cached = len(cache)
        print(f"[val] support cache: {n_cached}/{len(self.val_classes)} classes, "
              f"k_shot={self.k_shot}")
        return cache

    @torch.no_grad()
    def _validate(
        self, support_cache: dict[int, tuple[torch.Tensor, torch.Tensor]]
    ) -> dict[str, float]:
        """在验证集子集上评估 mIoU 和 FB-IoU.
        Evaluate mIoU and FB-IoU on a validation subset.

        :return: {"val/mIoU": float, "val/FB-IoU": float, "val/pixel_acc": float}
        """
        was_training = self.model.training
        self.model.eval()

        # Sample val tiles (fixed seed for reproducibility across epochs)
        val_rng = random.Random(self.seed + 2000)
        val_tiles = list(range(len(self.val_ds)))
        if len(val_tiles) > self.val_samples:
            val_tiles = val_rng.sample(val_tiles, self.val_samples)

        # Accumulators for mIoU
        cls_inter = defaultdict(float)
        cls_union = defaultdict(float)
        # Accumulators for FB-IoU
        fg_inter = 0.0
        fg_union = 0.0
        bg_inter = 0.0
        bg_union = 0.0
        pixel_correct = 0.0
        pixel_total = 0.0

        for idx in tqdm(val_tiles, desc="validate", leave=False):
            sample = self.val_ds[idx]
            emb = self._embed(sample["image"])

            # Aggregate all-class FG prediction and GT for FB-IoU
            all_fg_pred = np.zeros((256, 256), dtype=bool)
            all_fg_gt = np.zeros((256, 256), dtype=bool)

            for cls in self.val_classes:
                sup_data = support_cache.get(cls)
                if sup_data is None:
                    continue
                sup_feat, sup_mask = sup_data

                # GT mask for this class on this tile
                gt = np.zeros((256, 256), dtype=bool)
                for inst in sample["instances"]:
                    if inst["category_id"] == cls:
                        gt = gt | inst["mask"].numpy()

                # Predict
                try:
                    masks_pred, scores = self.model.predict(
                        emb, sup_feat, sup_mask,
                        (1024, 1024), (256, 256), score_thr=0.3,
                    )
                    # Merge all instance masks into one FG mask
                    if len(masks_pred) > 0:
                        pred = masks_pred.cpu().numpy().any(axis=0)
                    else:
                        pred = np.zeros((256, 256), dtype=bool)
                except (RuntimeError, ValueError, IndexError) as exc:
                    print(f"[WARN] prediction failed for tile {idx} class {cls}: {exc}")
                    pred = np.zeros((256, 256), dtype=bool)

                # Per-class IoU
                inter = float((pred & gt).sum())
                union = float((pred | gt).sum())
                cls_inter[cls] += inter
                cls_union[cls] += union

                # FB-IoU accumulation
                all_fg_pred = all_fg_pred | pred
                all_fg_gt = all_fg_gt | gt

            # FB-IoU per tile
            fg_inter += float((all_fg_pred & all_fg_gt).sum())
            fg_union += float((all_fg_pred | all_fg_gt).sum())
            bg_pred = ~all_fg_pred
            bg_gt = ~all_fg_gt
            bg_inter += float((bg_pred & bg_gt).sum())
            bg_union += float((bg_pred | bg_gt).sum())
            pixel_correct += float((all_fg_pred == all_fg_gt).sum())
            pixel_total += float(all_fg_gt.size)

        # Compute mIoU
        per_class_iou = {}
        for cls in self.val_classes:
            if cls_union[cls] > 0:
                per_class_iou[cls] = cls_inter[cls] / cls_union[cls]
            else:
                per_class_iou[cls] = float("nan")
        valid_ious = [v for v in per_class_iou.values() if not math.isnan(v)]
        miou = float(np.mean(valid_ious)) if valid_ious else 0.0

        # Compute FB-IoU
        n_components = 0
        fb_iou = 0.0
        if fg_union > 0:
            fb_iou += (fg_inter / fg_union)
            n_components += 1
        if bg_union > 0:
            fb_iou += (bg_inter / bg_union)
            n_components += 1
        fb_iou = fb_iou / n_components if n_components > 0 else 0.0

        pixel_acc = pixel_correct / max(pixel_total, 1.0)

        if was_training:
            self.model.train()

        return {
            "val/mIoU": round(miou, 4),
            "val/FB-IoU": round(fb_iou, 4),
            "val/pixel_acc": round(pixel_acc, 4),
            "val/n_classes": len(valid_ious),
        }

    # ── 主训练循环 | Main Training Loop ──

    def train(self) -> Path:
        """运行完整训练流程 | Run full training pipeline.

        每 epoch: 训练 N 个 episode → 验证 (按间隔) → checkpoint → 日志。
        Each epoch: train N episodes → validate (at interval) → checkpoint → log.
        """
        # 构建固定验证 support cache (FSS 标准协议)
        # Build fixed validation support cache (FSS protocol)
        print("[val] building support cache ...")
        val_cache = self._build_val_support_cache()
        if not val_cache:
            print("[val] WARNING: empty support cache! Validation disabled.")

        self.model.train()
        best_miou = -1.0
        best_loss = float("inf")
        best_path = self.out_dir / "best_model.pt"
        val_history: list[dict] = []

        for epoch in range(self.epochs):
            # ── 训练阶段 | Training ──
            agg: dict[str, float] = {}
            # 聚合本 epoch 所有 episode 的指标 | Aggregate metrics across episodes
            n = 0
            pbar = tqdm(range(self.episodes_per_epoch), desc=f"epoch {epoch}")
            for _ in pbar:
                metrics = self._train_episode(self.sampler.sample())
                if metrics is None:
                    continue
                n += 1
                for k, v in metrics.items():
                    agg[k] = agg.get(k, 0.0) + v
                pbar.set_postfix(loss=f"{metrics['loss']:.3f}")
            self.scheduler.step()  # 余弦退火: 每个 epoch 一步 | cosine annealing: one step per epoch

            mean = {k: v / max(n, 1) for k, v in agg.items()}
            current_lr = self.optimizer.param_groups[0]["lr"]
            mean["lr"] = current_lr
            for k, v in mean.items():
                self.logger.log_metric(f"train/{k}", v, step=epoch, phase="train")

            # ── 验证阶段 | Validation ──
            val_metrics: dict[str, float] = {}
            if val_cache and (epoch % self.val_every == 0 or epoch == self.epochs - 1):
                val_metrics = self._validate(val_cache)
                for k, v in val_metrics.items():
                    self.logger.log_metric(k, v, step=epoch, phase="val")

            # ── 日志记录 | Logging ──
            val_str = ""
            if val_metrics:
                val_str = (f"val_mIoU={val_metrics.get('val/mIoU', 0):.4f} "
                           f"val_FB={val_metrics.get('val/FB-IoU', 0):.4f} "
                           f"val_pix={val_metrics.get('val/pixel_acc', 0):.4f}")
            self.logger.log_info(
                "epoch",
                f"epoch {epoch:>3d}: loss={mean.get('loss', 0):.4f} "
                f"dice={mean.get('dice', 0):.4f} n={n} lr={current_lr:.2e} | {val_str}",
                step=epoch,
            )

            # ── Checkpoint 保存 | Save Checkpoint ──
            epoch_metrics = {**mean, **val_metrics}
            self._save(self.out_dir / "last_model.pt", epoch, epoch_metrics)

            # 主选择: 按 val/mIoU 选最优 (有验证时); 退化为按训练 loss 选
            # Primary selection: by val/mIoU (when validation runs); fallback = training loss
            if val_metrics:
                current_miou = val_metrics.get("val/mIoU", -1.0)
                if current_miou > best_miou:
                    best_miou = current_miou
                    self._save(best_path, epoch, epoch_metrics)
                    self.logger.log_info("best",
                        f"new best: val_mIoU={best_miou:.4f} (epoch {epoch})",
                        step=epoch)
            else:
                # 退化策略: 无验证 → 按训练 loss 选最优 | Fallback: no val → use training loss
                current_loss = mean.get("loss", float("inf"))
                if current_loss < best_loss:
                    best_loss = current_loss
                    self._save(best_path, epoch, epoch_metrics)
                    self.logger.log_info("best",
                        f"new best: loss={best_loss:.4f} (epoch {epoch})", step=epoch)

            val_history.append({"epoch": epoch, **epoch_metrics})

        # 保存验证历史记录 | Save validation history
        (self.out_dir / "val_history.json").write_text(
            json.dumps(val_history, indent=2), encoding="utf-8"
        )

        self.logger.flush()
        return best_path

    def _save(self, path: Path, epoch: int, metrics: dict) -> None:
        """保存检查点 (模型 + 优化器 + 配置 + 指标) | Save checkpoint (model + optimizer + config + metrics)."""
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.cfg,
            "metrics": metrics,
            "fold": self.fold,
            "k_shot": self.k_shot,
            "mode": self.mode,
        }
        if self.cat_adapter is not None:
            ckpt["cat_adapter"] = self.cat_adapter.state_dict()
        torch.save(ckpt, path)
        (self.out_dir / "last_metrics.json").write_text(
            json.dumps({"epoch": epoch, **metrics}, indent=2), encoding="utf-8"
        )


# ═══════════════════════════════════════════════════════════════════
# CLI — 命令行接口 | Command-Line Interface
# ═══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """解析命令行参数 | Parse CLI arguments.

    所有参数都可选 — 未指定时使用配置文件默认值。
    All args are optional — config file defaults are used when not specified.
    """
    p = argparse.ArgumentParser(description="AdaSAM iSAID-5i Few-shot Training")
    p.add_argument("--config", default=str(_REPO_ROOT / "configs" / "isaid_5i.yaml"))
    p.add_argument("--fold", type=int, default=None, help="fold 0/1/2")
    p.add_argument("--k-shot", type=int, default=None)
    p.add_argument("--mode", default=None, choices=["base", "novel", "all"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--episodes", "--steps", type=int, default=None, dest="steps")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--weights", default=None, help="MobileSAM weights path override")
    return p.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    """加载 YAML 配置并应用 CLI 覆盖 | Load YAML config and apply CLI overrides.

    CLI 参数优先级高于配置文件。以 `--config` 指定的文件为基础,
    用 `--fold`, `--k-shot` 等参数覆盖对应字段。
    CLI args take precedence over config file values.
    """
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # CLI 参数覆盖映射 | CLI → config key path mapping
    overrides = [
        (("data", "fold"), args.fold),
        (("fewshot", "k_shot"), args.k_shot),
        (("fewshot", "train_mode"), args.mode),
        (("train", "epochs"), args.epochs),
        (("train", "episodes_per_epoch"), args.steps),
        (("train", "lr"), args.lr),
        (("train", "device"), args.device),
        (("seed",), args.seed),
        (("output_dir",), args.output_dir),
        (("data", "data_root"), args.data_root),
    ]
    for keys, val in overrides:
        if val is not None:
            d = cfg
            for k in keys[:-1]:
                d = d.setdefault(k, {})  # 自动创建缺失的嵌套键 | auto-create missing nested keys
            d[keys[-1]] = val
    if args.weights is not None:
        cfg.setdefault("backbone", {})["checkpoint"] = args.weights
    return cfg


def main() -> None:
    """主入口 | Main entry point.

    解析参数 → 加载配置 → 创建训练器 → 开始训练。
    Parse args → load config → create trainer → start training.
    """
    args = parse_args()
    cfg = load_config(args)
    trainer = ISAID5iTrainer(cfg, args)
    best = trainer.train()
    print(f"\n[train_isaid_5i] done. best: {best}")


if __name__ == "__main__":
    main()
