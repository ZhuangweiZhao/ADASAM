"""
AdaSAM Stage 2 — Few-shot Semantic Learning.
=============================================

Stage 2: 加载 Stage 1 Domain Adapter (冻结) → Episode 训练 SPG + GeometricPrior +
PromptFusion + SAM Decoder。Novel 类直接推理, 不再训练。

Loads Stage 1 adapter (frozen) → Episode training of SPG + GeometricPrior +
PromptFusion + SAM Decoder. Novel classes inferred directly (no finetune).

用法 | Usage::

    # 完整训练 (需先完成 Stage 1)
    python tools/adasam/train_stage2.py --fold 0 --k-shot 5 --epochs 50 \\
        --stage1-ckpt runs/stage1_fold0_seed42/best_adapter.pt

    # 烟测试
    python tools/adasam/train_stage2.py --fold 0 --k-shot 5 --epochs 1 --steps 5 \\
        --stage1-ckpt runs/stage1_fold0_seed42/best_adapter.pt
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
_REPO_ROOT = Path(__file__).resolve().parents[2]
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
from adasam.losses import SemanticSegLoss
from adasam.model import AdaSAMModel, AdaSAMModelConfig
from adasam.utils import set_seed
from adasam.utils.debug_trace import configure_from_config, tracer
from adasam.utils.transforms import preprocess_image, resize_mask

# Import new prompt modules for type hints
from adasam.prompt import SPGOutput


# ═══════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════

class ISAID5iTrainer:
    """iSAID-5i 小样本训练器 | Few-shot trainer for iSAID-5i."""

    def __init__(self, cfg: dict, args: argparse.Namespace) -> None:
        """初始化 Stage 2 训练器 | Initialize Stage 2 trainer.

        完成以下设置: 随机种子/设备、数据集+Episode Sampler、加载 Stage 1 Adapter、
        模型+Backbone、Loss、优化器+学习率调度器、输出目录+日志。
        Sets up: random seed/device, dataset + episode sampler, load Stage 1 adapter,
        model + backbone, loss, optimizer + scheduler, output dir + logger.
        """
        self.cfg = cfg
        self.args = args
        self.stage1_ckpt_path: Path = Path(args.stage1_ckpt)
        self.seed = int(cfg.get("seed", 42))
        set_seed(self.seed)
        self.device = torch.device(
            cfg["train"].get("device", "cuda") if torch.cuda.is_available() else "cpu"
        )
        self._rng = random.Random(self.seed)

        # ── 数据 | Data ──
        self.fold = int(cfg["data"].get("fold", 0))
        self.k_shot = int(cfg["fewshot"].get("k_shot", 5))
        data_root = self._resolve(cfg["data"]["data_root"])

        # Stage 2 trains on base classes only (novel → direct inference, no training)
        self.mode = "base"
        self.train_ds = ISAID5iDataset(root=data_root, fold=self.fold, split="train", mode=self.mode)
        self.train_classes = self.train_ds.visible_classes()
        print(f"[Stage2] fold={self.fold} mode={self.mode} classes={self.train_classes}")
        for cls in self.train_classes:
            n = len(self.train_ds.class_to_tiles(cls))
            name = ISAID5I_CATEGORIES.get(cls, f"cls{cls}")
            print(f"  class {cls:>2d} ({name:<20s}): {n} tiles")

        # Episode Sampler
        self.sampler = ISAID5iEpisodeSampler(
            self.train_ds, k_shot=self.k_shot, seed=self.seed,
            min_tiles=int(cfg["fewshot"].get("min_tiles", 10)),
        )
        self.eligible = self.sampler.eligible_classes()
        print(f"[Stage2] eligible classes after filtering: {len(self.eligible)}")

        # ── 验证集 | Validation ──
        tcfg = cfg["train"]
        self.val_every = int(tcfg.get("val_every", 10))
        self.val_samples = int(tcfg.get("val_samples", 30))
        self.val_ds = ISAID5iDataset(root=data_root, fold=self.fold, split="val", mode=self.mode)
        self.val_classes = self.val_ds.visible_classes()
        print(f"[Stage2] val classes: {len(self.val_classes)}, "
              f"val_every={self.val_every}, val_samples={self.val_samples}")

        # ── 模型 | Model ──
        ckpt_path = self._resolve(cfg["backbone"]["checkpoint"])
        sam = build_mobile_sam(ckpt_path, cfg["backbone"].get("model_type", "vit_t"), self.device)
        self.backbone = MobileSAMBackbone(sam.image_encoder, sam.image_encoder.img_size).to(self.device)
        self.image_size = self.backbone.img_size
        self.embed_dim = int(cfg.get("support_encoder", {}).get("embed_dim", 256))
        self.model = AdaSAMModel(sam, AdaSAMModelConfig.from_dict(cfg)).to(self.device)
        self.num_probes = self.model.num_probes

        # ── Stage 1 Adapter: 加载并冻结 | Load and freeze ──
        if not self.stage1_ckpt_path.exists():
            raise FileNotFoundError(f"Stage 1 checkpoint not found: {self.stage1_ckpt_path}")
        self.cat_adapter = self._load_stage1_adapter(self.stage1_ckpt_path)

        # ── 优化器 & 学习率调度 | Optimizer & LR Scheduler ──
        self.grad_clip = float(tcfg.get("grad_clip", 1.0))
        self.epochs = int(tcfg.get("epochs", 50))
        self.episodes_per_epoch = int(tcfg.get("episodes_per_epoch", 200))

        lr = float(cfg["train"].get("lr", 1e-4))
        param_groups = self._build_param_groups(cfg, lr)
        self._trainable = [p for g in param_groups for p in g["params"]]
        self.optimizer = AdamW(
            param_groups, lr=lr, weight_decay=float(tcfg.get("weight_decay", 1e-4))
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        # ── 损失函数 | Criterion ──
        loss_cfg = cfg.get("loss", {})
        self.criterion = SemanticSegLoss(
            prior_weight=float(loss_cfg.get("prior_weight", 0.3)),
            reg_weight=float(loss_cfg.get("reg_weight", 0.0)),
            focal_weight=float(loss_cfg.get("focal_weight", 1.0)),
            dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
            focal_gamma=float(loss_cfg.get("focal_gamma", 5.0)),
            focal_eps=float(loss_cfg.get("focal_eps", 1e-4)),
        )

        # ── 输出目录 & 日志 | Output & Logging ──
        exp = f"stage2_fold{self.fold}_k{self.k_shot}_seed{self.seed}"
        self.out_dir = self._resolve(cfg.get("output_dir", "runs")) / exp
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # ── Debug tracer ──
        configure_from_config(cfg, output_dir=self.out_dir)
        self.logger = get_logger("trainer.stage2")
        if not self.logger.backends:
            self.logger.add_backend(ConsoleBackend())
            self.logger.add_backend(FileBackend(str(self.out_dir / "train.jsonl")))

        n_train = sum(p.numel() for p in self._trainable) / 1e6
        init_info = (
            f"stage2 fold={self.fold} k={self.k_shot} "
            f"device={self.device} trainable={n_train:.2f}M probes={self.num_probes} "
            f"classes={self.eligible} out={self.out_dir} "
            f"stage1_ckpt={self.stage1_ckpt_path}"
        )
        self.logger.log_info("init", init_info)

    @staticmethod
    def _resolve(path: str | Path) -> Path:
        """将相对路径转为相对于 repo 根目录的绝对路径 | Resolve relative path to repo root."""
        p = Path(path)
        return p if p.is_absolute() else (_REPO_ROOT / p)

    # ── Stage 1 Adapter 加载 | Stage 1 Adapter Loading ──

    def _load_stage1_adapter(self, ckpt_path: Path) -> CATAdapter:
        """从 Stage 1 checkpoint 加载 Domain Adapter 并冻结。

        Load domain-adapted CATAdapter from Stage 1 checkpoint and freeze it.
        Stage 1 adapter provides domain-aware feature initialization for Stage 2.

        :return: CATAdapter loaded from Stage 1 (frozen).
        """
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        adapter_state = ckpt.get("adapter")
        if adapter_state is None:
            raise KeyError(f"Stage 1 checkpoint {ckpt_path} has no 'adapter' key. "
                           f"Keys: {list(ckpt.keys())}")

        adapter_cfg = ckpt.get("config", {}).get("adapter", {})
        adapter = CATAdapter(
            dim=self.embed_dim,
            bottleneck=int(adapter_cfg.get("bottleneck", 64)),
        ).to(self.device)
        adapter.load_state_dict(adapter_state)
        # Freeze: adapter weights are fixed during Stage 2
        for p in adapter.parameters():
            p.requires_grad_(False)
        adapter.eval()

        s1_epoch = ckpt.get("epoch", "?")
        s1_fold = ckpt.get("fold", "?")
        print(f"[load_stage1] adapter from: {ckpt_path}")
        print(f"  stage1 epoch={s1_epoch} fold={s1_fold} "
              f"params={sum(p.numel() for p in adapter.parameters()):,}")
        return adapter

    # ── 优化器参数分组 | Optimizer Param Groups ──

    def _build_param_groups(self, cfg: dict, lr: float) -> list[dict]:
        """Stage 2 参数分组 | Stage 2 param groups.

        SPG + SupportEncoder + GeometricPrior + PromptFusion 全速,
        SAM MaskDecoder 低速 (×sam_decoder_lr_mult, 保留预训练知识).
        Adapter 已冻结, 不参与训练。

        SPG + SupportEncoder + GeometricPrior + PromptFusion full-rate,
        SAM MaskDecoder reduced-rate.

        :return: list of param_group dicts for AdamW.
        """
        tcfg = cfg["train"]
        sam_mult = float(tcfg.get("sam_decoder_lr_mult", 0.1))

        groups = [
            {"params": list(self.model.spg.parameters()), "lr": lr},
            {"params": list(self.model.support_encoder.parameters()), "lr": lr},
            {"params": [
                p for p in self.model.sam_decoder.mask_decoder.parameters()
                if p.requires_grad
            ], "lr": lr * sam_mult},
        ]
        if self.model.geometric_prior is not None:
            groups.append({"params": list(self.model.geometric_prior.parameters()), "lr": lr})
        if self.model.prompt_fusion is not None:
            groups.append({"params": list(self.model.prompt_fusion.parameters()), "lr": lr})

        print(f"[params] stage2: lr={lr}, sam_lr={lr * sam_mult:.1e}, "
              f"groups={len(groups)}")
        return groups

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
            fg = self._class_foreground(idx, class_id)
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

    def _class_foreground(self, index: int, class_id: int) -> torch.Tensor | None:
        """获取指定 tile 上某类的合并前景 mask | Get merged FG mask for a class on a tile.

        :return: [H,W] float tensor 或 None (该类别不存在于该 tile).
        """
        return self.train_ds.get_class_mask(index, class_id)

    # ── 单 Episode 训练 | Single Episode Training ──

    def _train_episode(self, episode: dict) -> dict | None:
        """执行一个 episode 的前向+反向 | Run one episode: forward + backward.

        流程: support memory → query embedding → SPG → SAM Decoder → loss → backward.
        Flow: support memory → query embedding → SPG → SAM Decoder → loss → backward.

        :return: loss 指标字典, 或 None (episode 无效 | invalid episode).
        """
        cls = episode["class_id"]

        # 构建 support 特征 (K 张 support tile) | Build support features (K support tiles)
        support_data = self._build_support_memory(episode["support_indices"], cls)
        if support_data is None:
            return None
        support_features, support_masks_grid = support_data

        # 提取 query GT mask (语义: 同类合并为 binary FG mask)
        # Extract query GT as binary FG mask (semantic: class-level merge)
        query = self.train_ds[episode["query_index"]]
        gt_binary = self.train_ds.get_class_mask(episode["query_index"], cls)
        if gt_binary is None or gt_binary.sum() < 1:
            return None  # 无前景 | no foreground
        gt_binary = gt_binary.to(self.device)

        # 前向传播 + 损失计算 | Forward + loss
        emb = self._embed(query["image"])
        spg_out, low_res, iou_pred = self.model.forward_train(
            emb, support_features, support_masks_grid
        )

        # low_res is [1, 1, 256, 256] (single mask from single token)
        fg_logits = low_res[0, 0]   # [256, 256]
        bg_logits = -fg_logits       # approximate BG
        pred_2ch = torch.stack([bg_logits, fg_logits], dim=0).unsqueeze(0)  # [1, 2, 256, 256]

        # Gather SPG unified prior masks for deep supervision (L_prior)
        # prior_aux now stores unified [1, gh, gw] masks (not per-probe [N, gh, gw])
        prior_masks = []
        for aux_entry in spg_out.prior_aux:
            prior_masks.append(aux_entry["prior_mask"])  # [1, gh, gw]

        losses = self.criterion(
            pred_2ch, gt_binary.unsqueeze(0),
            prior_masks=prior_masks,
            prior_mask=spg_out.prior_mask,
        )

        # 反向传播 | Backward
        self.optimizer.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self._trainable, self.grad_clip)

        # ── Debug: gradient trace ──
        if tracer.should_log and tracer.level >= 3:
            tracer.grad("spatial_prompt_scale", self.model.spatial_prompt_scale)
            tracer.grad_summary(self.model.spg, prefix="SPG")
            tracer.section("AdaSAM — Post-backward gradient check")

        self.optimizer.step()

        # ── Debug: advance step counter ──
        tracer.step()

        metrics = {
            "loss": float(losses["loss"].detach()),
            "L_main": float(losses["L_main"]),
            "L_prior": float(losses["L_prior"]),
            "L_reg": float(losses["L_reg"]),
            "main_ce": float(losses["main_ce"]),
            "main_focal": float(losses["main_focal"]),
            "main_dice": float(losses["main_dice"]),
        }
        return metrics

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
        # Diagnostics: track prediction quality
        n_total_calls = 0        # total predict() calls
        n_nonempty = 0           # predict() returned ≥1 mask
        n_score_filtered = 0     # all masks filtered by score threshold
        n_exception = 0          # predict() raised exception
        score_sum = 0.0          # sum of max scores (for averaging)
        area_sum = 0.0           # sum of prediction areas

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

                # GT mask for this class on this tile (semantic: class-level)
                gt_mask = self.val_ds.get_class_mask(idx, cls)
                gt = gt_mask.numpy().astype(bool) if gt_mask is not None else np.zeros((256, 256), dtype=bool)

                # Predict
                n_total_calls += 1
                try:
                    masks_pred, scores = self.model.predict(
                        emb, sup_feat, sup_mask,
                        (1024, 1024), (256, 256), score_thr=0.1,
                    )
                    # Single mask output [1, H, W] or empty [1, H, W]
                    if masks_pred.shape[0] > 0 and masks_pred.sum() > 0:
                        pred = masks_pred.cpu().numpy().squeeze(0)
                        n_nonempty += 1
                        area_sum += float(pred.sum())
                    else:
                        pred = np.zeros((256, 256), dtype=bool)
                        n_score_filtered += 1
                    if len(scores) > 0:
                        score_sum += float(scores.max())
                except (RuntimeError, ValueError, IndexError) as exc:
                    print(f"[WARN] prediction failed for tile {idx} class {cls}: {exc}")
                    pred = np.zeros((256, 256), dtype=bool)
                    n_exception += 1

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

        # ── Validation Diagnostics ──
        pct_nonempty = 100.0 * n_nonempty / max(n_total_calls, 1)
        pct_filtered = 100.0 * n_score_filtered / max(n_total_calls, 1)
        pct_exc = 100.0 * n_exception / max(n_total_calls, 1)
        avg_score = score_sum / max(n_nonempty, 1)
        avg_area = area_sum / max(n_nonempty, 1)
        print(f"\n[val diag] calls={n_total_calls} | "
              f"nonempty={n_nonempty} ({pct_nonempty:.1f}%) | "
              f"score_filtered={n_score_filtered} ({pct_filtered:.1f}%) | "
              f"exception={n_exception} ({pct_exc:.1f}%)")
        print(f"[val diag] avg_max_score={avg_score:.4f} | "
              f"avg_pred_area={avg_area:.0f} px | "
              f"valid_classes={len(valid_ious)}")
        # Per-class detail
        for cls in sorted(self.val_classes):
            inter = cls_inter[cls]
            union = cls_union[cls]
            iou = per_class_iou[cls]
            iou_str = f"{iou:.4f}" if not math.isnan(iou) else "NaN"
            print(f"[val diag]   cls {cls:>2d}: inter={inter:.0f} union={union:.0f} IoU={iou_str}")

        if was_training:
            self.model.train()

        return {
            "val/mIoU": round(miou, 4),
            "val/FB-IoU": round(fb_iou, 4),
            "val/pixel_acc": round(pixel_acc, 4),
            "val/n_classes": len(valid_ious),
        }

    @torch.no_grad()
    def _save_aux_viz(
        self,
        support_cache: dict[int, tuple[torch.Tensor, torch.Tensor]],
        epoch: int,
        n_samples: int = 4,
    ) -> None:
        """保存 Prior 输出可视化 | Save prior output visualizations.

        每 val_every epoch 保存几张对比图:
        [query image | GT union | prior_mask | main prediction]
        用于诊断 SPG prior 是否学到了有意义的结构。
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        was_training = self.model.training
        self.model.eval()

        out_dir = self.out_dir / "aux_viz"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 固定采样 tile (同 epoch 可比) | fixed tile sampling for comparability
        viz_rng = random.Random(self.seed + 3000)
        pool = [i for i in range(len(self.val_ds))
                if len(self.val_ds[i]["regions"]) > 0]
        if len(pool) > n_samples:
            pool = viz_rng.sample(pool, n_samples)

        fig, axes = plt.subplots(len(pool), 4, figsize=(16, 4 * len(pool)))
        if len(pool) == 1:
            axes = axes[None, :]  # ensure 2D

        for row, idx in enumerate(pool):
            sample = self.val_ds[idx]
            emb = self._embed(sample["image"])

            # 选该 tile 上 GT 实例最多的类别 | pick class with most GT instances
            best_cls = max(self.val_classes, key=lambda c:
                sum(1 for i in sample["regions"] if i["category_id"] == c))
            sup_data = support_cache.get(best_cls)
            if sup_data is None:
                sup_data = next(iter(support_cache.values()))
            sup_feat, sup_mask = sup_data

            # 运行一次前向获取 prior_mask | run forward once to get prior_mask
            try:
                spg_out, low_res, iou_pred = self.model.forward_train(
                    emb, sup_feat, sup_mask
                )
                masks_pred, scores = self.model.predict(
                    emb, sup_feat, sup_mask,
                    (1024, 1024), (256, 256), score_thr=0.1,
                )
            except (RuntimeError, ValueError, IndexError):
                continue

            # --- 1) Query image ---
            img = sample["image"].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            axes[row, 0].imshow(img)
            axes[row, 0].set_title(f"Query (cls={best_cls})", fontsize=9)
            axes[row, 0].axis("off")

            # --- 2) GT union (semantic: class-level mask) ---
            gt_m = self.val_ds.get_class_mask(idx, best_cls)
            gt = gt_m.numpy().astype(bool) if gt_m is not None else np.zeros(img.shape[:2], dtype=bool)
            axes[row, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
            axes[row, 1].set_title("GT Union", fontsize=9)
            axes[row, 1].axis("off")

            # --- 3) prior_mask (SPG prior projection) ---
            if spg_out.prior_mask is not None:
                pm = spg_out.prior_mask[0, 0].sigmoid().cpu().numpy()
                axes[row, 2].imshow(pm, cmap="viridis", vmin=0, vmax=1)
                axes[row, 2].set_title(f"prior_mask\nμ={pm.mean():.3f} σ={pm.std():.3f}", fontsize=9)
            else:
                axes[row, 2].text(0.5, 0.5, "N/A", ha="center", va="center",
                                  transform=axes[row, 2].transAxes, fontsize=12)
            axes[row, 2].axis("off")

            # --- 4) Main prediction (merged) ---
            if len(masks_pred) > 0:
                pred = masks_pred.cpu().numpy().any(axis=0)
            else:
                pred = np.zeros(img.shape[:2], dtype=bool)
            axes[row, 3].imshow(pred, cmap="gray", vmin=0, vmax=1)
            axes[row, 3].set_title("Main Prediction", fontsize=9)
            axes[row, 3].axis("off")

        plt.tight_layout()
        save_path = out_dir / f"epoch_{epoch:03d}.png"
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"[aux_viz] saved to {save_path}")

        if was_training:
            self.model.train()

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
                # 保存 Aux/Prompt 可视化 (每 val_every epoch) | Save aux/prompt viz
                self._save_aux_viz(val_cache, epoch)

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
        tracer.summary()
        tracer.close()
        return best_path

    def _save(self, path: Path, epoch: int, metrics: dict) -> None:
        """保存检查点 | Save checkpoint (model + optimizer + config + metrics)."""
        ckpt = {
            "epoch": epoch,
            "stage": "stage2",
            "mode": self.mode,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.cfg,
            "metrics": metrics,
            "fold": self.fold,
            "k_shot": self.k_shot,
            "visible_classes": self.train_classes,
            "stage1_ckpt": str(self.stage1_ckpt_path),
        }
        if self.cat_adapter is not None:
            ckpt["cat_adapter"] = self.cat_adapter.state_dict()
        torch.save(ckpt, path)
        (self.out_dir / "last_metrics.json").write_text(
            json.dumps({"epoch": epoch, "stage": "stage2", **metrics}, indent=2), encoding="utf-8"
        )


# ═══════════════════════════════════════════════════════════════════
# CLI — 命令行接口 | Command-Line Interface
# ═══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """解析命令行参数 | Parse CLI arguments.

    所有参数都可选 — 未指定时使用配置文件默认值。
    All args are optional — config file defaults are used when not specified.
    """
    p = argparse.ArgumentParser(description="AdaSAM Stage 2: Few-shot Semantic Learning")
    p.add_argument("--config", default=str(_REPO_ROOT / "configs" / "isaid_5i.yaml"))
    p.add_argument("--stage1-ckpt", required=True,
                   help="path to Stage 1 adapter checkpoint (best_adapter.pt)")
    p.add_argument("--fold", type=int, default=None, help="fold 0/1/2")
    p.add_argument("--k-shot", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--episodes", "--steps", type=int, default=None, dest="steps")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--weights", default=None, help="MobileSAM weights path override")
    p.add_argument("--debug", type=int, default=None, choices=[0, 1, 2, 3],
                   help="enable data-flow debug trace (0=off, 1=shape, 2=+spatial, 3=+grad)")
    p.add_argument("--val-every", type=int, default=None,
                   help="validate every N epochs (default: 10)")
    p.add_argument("--val-samples", type=int, default=None,
                   help="validation tile samples (default: 30)")
    p.add_argument("--prior-weight", type=float, default=None,
                   help="L_prior loss weight (default: 0.3, set 0 to disable)")
    return p.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    """加载 YAML 配置并应用 CLI 覆盖 | Load YAML config and apply CLI overrides.

    CLI 参数优先级高于配置文件。以 `--config` 指定的文件为基础,
    用 `--fold`, `--k-shot` 等参数覆盖对应字段。
    CLI args take precedence over config file values.
    """
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Stage 2 always uses base classes (mode="base")
    cfg.setdefault("fewshot", {})["train_mode"] = "base"

    # CLI 参数覆盖映射 | CLI → config key path mapping
    overrides = [
        (("data", "fold"), args.fold),
        (("fewshot", "k_shot"), args.k_shot),
        (("train", "epochs"), args.epochs),
        (("train", "episodes_per_epoch"), args.steps),
        (("train", "lr"), args.lr),
        (("train", "val_every"), args.val_every),
        (("train", "val_samples"), args.val_samples),
        (("train", "device"), args.device),
        (("seed",), args.seed),
        (("output_dir",), args.output_dir),
        (("data", "data_root"), args.data_root),
        (("loss", "prior_weight"), args.prior_weight),
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

    # --stage1-ckpt is required
    if not Path(args.stage1_ckpt).exists():
        print(f"ERROR: Stage 1 checkpoint not found: {args.stage1_ckpt}")
        print("  Run Stage 1 first: python tools/adasam/train_stage1.py --fold 0 --epochs 50")
        sys.exit(1)

    cfg = load_config(args)
    # CLI --debug 覆盖 yaml debug.enabled
    if args.debug is not None:
        cfg.setdefault("debug", {})["enabled"] = True
        cfg["debug"]["level"] = args.debug
        cfg["debug"]["log_every"] = 1
    trainer = ISAID5iTrainer(cfg, args)
    best = trainer.train()
    print(f"\n[train_isaid_5i] done. best: {best}")


if __name__ == "__main__":
    main()
