"""
训练流程 | Training pipeline.
============================

AdaSAM 的**唯一**训练入口 (单一 Trainer 类, 无 if-mode 分支)。
The single training entry point of AdaSAM (one Trainer class, no if-mode branches).

范式 | Paradigm — Hungarian set prediction (Mask2Former 式):
    - 冻结 MobileSAM 图像编码器; 训练 DensePromptGenerator + SAM MaskDecoder
      (+ 可选 CATAdapter)。Frozen MobileSAM encoder; DensePromptGenerator +
      SAM MaskDecoder (+ optional CATAdapter) train.
    - 每个 episode: K 张 support → 类原型 (仅语义条件); query 特征 + 原型 →
      DPG 生成 N 个实例查询 → SAM 解码 N 个掩码 → 与该类全部 GT 实例做匈牙利
      匹配 → focal+dice+objectness+IoU-head 监督 (含 DPG 逐层深监督)。
      Per episode: prototype from K supports (semantic condition only); DPG
      generates N instance queries from query features + prototype → SAM
      decodes N masks → Hungarian matching against all class GT instances →
      focal+dice+objectness+IoU-head supervision (with DPG deep supervision).
    - 无点提示、无框提示、无 top-k、无 NMS — 训练与推理共用 AdaSAMModel。
      No point/box prompts, no top-k, no NMS — training and inference share
      the same AdaSAMModel forward.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from adasam.adapters import CATAdapter
from adasam.backbone import build_mobile_sam, MobileSAMBackbone
from adasam.datasets import EpisodeSampler, ISAIDInstanceDataset
from adasam.logging import get_logger
from adasam.logging.backends import ConsoleBackend, FileBackend
from adasam.losses import CriterionConfig, HungarianMatcher, MatcherConfig, SetCriterion
from adasam.model import AdaSAMModel, AdaSAMModelConfig
from adasam.prototype import PrototypeBuilder
from adasam.utils import set_seed
from adasam.utils.transforms import preprocess_image

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Trainer:
    """AdaSAM 小样本训练器 | AdaSAM few-shot trainer.

    :param config: 合并后的配置字典 (configs/base.yaml + CLI 覆盖) | merged config dict.
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.seed = int(config.get("seed", 42))
        set_seed(self.seed)

        self.device = torch.device(config["train"].get("device", "cuda")
                                   if torch.cuda.is_available() else "cpu")
        self._rng = random.Random(self.seed)

        # ── 数据先加载 (COCO JSON 解析瞬时占用 ~2.5GB, 先于 CUDA 上下文以降低 RAM 峰值) ──
        # Load data FIRST: the COCO JSON parse peaks at ~2.5GB; doing it before the CUDA
        # context avoids stacking peaks (model ~1.5GB + json ~2.5GB) that trigger OOM-kill.
        self.tile_size = int(config["data"].get("tile_size", 896))
        self.k_shot = int(config["fewshot"].get("k_shot", 5))
        self.mode = config["fewshot"].get("train_mode", "novel")
        _ann = config["data"].get("train_ann_file")
        self.dataset = ISAIDInstanceDataset(
            root=self._resolve(config["data"]["data_root"]),
            split="train", fold=int(config["data"].get("fold", 0)), mode=self.mode,
            ann_file=self._resolve(_ann) if _ann else None,
        )
        self.sampler = EpisodeSampler(
            self.dataset, k_shot=self.k_shot, seed=self.seed,
            min_tiles=int(config["fewshot"].get("min_tiles", 30)),
        )

        # ── 模型装配 (一次构建 Sam, 分发到 backbone/model) | Assemble model (one Sam) ──
        ckpt = self._resolve(config["backbone"]["checkpoint"])
        sam = build_mobile_sam(ckpt, config["backbone"].get("model_type", "vit_t"), self.device)
        self.backbone = MobileSAMBackbone(sam.image_encoder, sam.image_encoder.img_size).to(self.device)
        self.image_size = self.backbone.img_size

        self.embed_dim = int(config.get("prototype", {}).get("embed_dim", 256))
        self.model = AdaSAMModel(sam, AdaSAMModelConfig.from_dict(config)).to(self.device)
        self.num_queries = self.model.num_queries
        self.proto_builder = PrototypeBuilder(self.embed_dim)

        # ── 匹配器 / 损失准则 | Matcher / criterion ──
        loss_cfg = config.get("loss", {})
        self.criterion = SetCriterion(
            HungarianMatcher(MatcherConfig.from_dict(loss_cfg)),
            CriterionConfig.from_dict(loss_cfg),
        )

        # ── 训练配置 | Train config ──
        tcfg = config["train"]
        self.epochs = int(tcfg.get("epochs", 50))
        self.episodes_per_epoch = int(tcfg.get("episodes_per_epoch", 200))
        self.grad_clip = float(tcfg.get("grad_clip", 1.0))

        # ── CAT-SAM Adapter (optional) | CAT-SAM 适配器 (可选) ──
        self.cat_adapter = None
        if bool(tcfg.get("use_cat_adapter", False)):
            adapter_cfg = tcfg.get("cat_adapter", {})
            self.cat_adapter = CATAdapter(
                dim=self.embed_dim,
                bottleneck=int(adapter_cfg.get("bottleneck", 64)),
            ).to(self.device)

        # ── 优化器: 全新模块全 lr, 预训练 MaskDecoder 降 lr | Optimizer param groups ──
        lr = float(tcfg.get("lr", 1e-4))
        sam_mult = float(tcfg.get("sam_decoder_lr_mult", 0.1))
        param_groups = [{"params": list(self.model.dpg.parameters()), "lr": lr}]
        if self.model.sam_decoder.proto_adapter is not None:
            param_groups.append(
                {"params": list(self.model.sam_decoder.proto_adapter.parameters()), "lr": lr}
            )
        param_groups.append({
            "params": [p for p in self.model.sam_decoder.mask_decoder.parameters()
                       if p.requires_grad],
            "lr": lr * sam_mult,
        })
        if self.cat_adapter is not None:
            param_groups.append({"params": list(self.cat_adapter.parameters()), "lr": lr})
        self._trainable = [p for g in param_groups for p in g["params"]]
        self.optimizer = AdamW(param_groups, lr=lr,
                               weight_decay=float(tcfg.get("weight_decay", 1e-4)))
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        # ── 输出 / 日志 | Output / logging ──
        exp = f"train_fold{config['data'].get('fold', 0)}_k{self.k_shot}_{self.mode}_seed{self.seed}"
        self.out_dir = self._resolve(config.get("output_dir", "runs")) / exp
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("trainer")
        if not self.logger.backends:
            self.logger.add_backend(ConsoleBackend())
            self.logger.add_backend(FileBackend(str(self.out_dir / "train.jsonl")))
        n_train = sum(p.numel() for p in self._trainable) / 1e6
        self.logger.log_info("init",
                             f"device={self.device}, trainable={n_train:.2f}M, "
                             f"queries={self.num_queries}, "
                             f"classes={self.sampler.eligible_classes()}, out={self.out_dir}")
        if self.cat_adapter is not None:
            self.logger.log_info("adapter",
                               f"CAT-Adapter: dim={self.embed_dim}, "
                               f"params={sum(p.numel() for p in self.cat_adapter.parameters()):,}")

    # ── 路径工具 | Path helper ──

    @staticmethod
    def _resolve(path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (_REPO_ROOT / p)

    # ── 嵌入 | Embedding ──

    def _embed(self, image: torch.Tensor) -> torch.Tensor:
        """tile 图像 [3,H,W]∈[0,1] → 图像嵌入 [1,256,64,64] (冻结骨干 + 可训练适配器)。
        Frozen-backbone embedding, adapted by the trainable CATAdapter if enabled."""
        x, _ = preprocess_image(image)
        emb = self.backbone(x.unsqueeze(0).to(self.device))["image_embedding"]
        if self.cat_adapter is not None:
            emb = self.cat_adapter(emb)
        return emb

    # ── 原型构建 (仅语义条件) | Prototype build (semantic condition only) ──

    def _build_prototype(self, support_indices: list[int], class_id: int) -> Optional[torch.Tensor]:
        """由 K 张 support 构建类原型 | Build class prototype from K supports.

        support 批量过冻结骨干, 再过可训练 CATAdapter (Siamese, 梯度经原型流入适配器)。
        Supports are batch-encoded by the frozen backbone, then pass the trainable
        CATAdapter (Siamese; gradients flow into the adapter through the prototype).

        :return: [256] L2 归一化原型, 无有效 support 时 None | prototype or None.
        """
        images, masks = [], []
        for idx in support_indices:
            sample = self.dataset[idx]
            fg = self._class_foreground(sample["instances"], class_id, self.tile_size)
            if fg is None:
                continue
            x, _ = preprocess_image(sample["image"])
            images.append(x.to(self.device))
            masks.append(fg)
        if not images:
            return None

        feats = self.backbone(torch.stack(images, dim=0))["image_embedding"]  # [K,256,64,64]
        if self.cat_adapter is not None:
            feats = self.cat_adapter(feats)
        prototype = self.proto_builder.build(
            [feats[k] for k in range(feats.shape[0])], masks
        )
        if float(prototype.detach().norm()) == 0.0:
            return None
        return prototype

    @staticmethod
    def _class_foreground(instances: list[dict], class_id: int, size: int) -> Optional[torch.Tensor]:
        """类前景并集掩码 [H,W] | union FG mask of a class, or None if absent."""
        fg = torch.zeros(size, size, dtype=torch.bool)
        found = False
        for inst in instances:
            if inst["category_id"] == class_id:
                fg |= inst["mask"]
                found = True
        return fg.float() if found else None

    # ── 单 episode 训练 | Single-episode training step ──

    def _train_episode(self, episode: dict) -> Optional[dict]:
        cls = episode["class_id"]

        # 1. 原型 (语义条件) | prototype (semantic condition)
        prototype = self._build_prototype(episode["support_indices"], cls)
        if prototype is None:
            return None

        # 2. query GT 实例 (该类全部, 超出查询数时随机截断) | class GT instances
        query = self.dataset[episode["query_index"]]
        gt_list = [i["mask"] for i in query["instances"] if i["category_id"] == cls]
        if not gt_list:
            return None
        if len(gt_list) > self.num_queries:
            gt_list = self._rng.sample(gt_list, self.num_queries)
        gt_masks = torch.stack([m.float() for m in gt_list], dim=0).to(self.device)  # [M,H,W]

        # 3. 前向 + 匹配损失 | forward + matched loss
        emb = self._embed(query["image"])                        # [1,256,64,64]
        dpg_out, low_res, iou_pred = self.model.forward_train(emb, prototype)
        losses = self.criterion(low_res[:, 0], iou_pred[:, 0], dpg_out, gt_masks)

        # 4. 反传 | backward
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
            "n_matched": float(losses["n_matched"]),
            "mean_obj_matched": float(losses["mean_obj_matched"]),
            "mean_obj_unmatched": float(losses["mean_obj_unmatched"]),
            "n_inst": gt_masks.shape[0],
        }

    # ── 主循环 | Main loop ──

    def train(self) -> Path:
        """运行训练, 返回最优 checkpoint 路径 | Run training, return best checkpoint path."""
        self.model.train()
        best_loss = float("inf")
        best_path = self.out_dir / "best_model.pt"

        for epoch in range(self.epochs):
            agg: dict[str, float] = {}
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
            epoch_lr = self.optimizer.param_groups[0]["lr"]      # 本轮实际 lr | lr used this epoch
            self.scheduler.step()

            mean = {k: v / max(n, 1) for k, v in agg.items()}
            mean["lr"] = epoch_lr
            for k, v in mean.items():
                self.logger.log_metric(f"train/{k}", v, step=epoch, phase="train")
            self.logger.log_info("epoch",
                                 f"epoch {epoch}: loss={mean.get('loss', 0):.4f} "
                                 f"dice={mean.get('dice', 0):.4f} "
                                 f"obj_matched={mean.get('mean_obj_matched', 0):.3f} "
                                 f"obj_unmatched={mean.get('mean_obj_unmatched', 0):.3f} "
                                 f"n={n}", step=epoch)

            self._save(self.out_dir / "last_model.pt", epoch, mean)
            if mean.get("loss", float("inf")) < best_loss:
                best_loss = mean["loss"]
                self._save(best_path, epoch, mean)

        self.logger.flush()
        return best_path

    # ── Checkpoint ──

    def _save(self, path: Path, epoch: int, metrics: dict) -> None:
        """保存 checkpoint (统一 schema, 无条件键) | Save checkpoint (uniform schema)."""
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.cfg,
            "metrics": metrics,
        }
        if self.cat_adapter is not None:
            ckpt["cat_adapter"] = self.cat_adapter.state_dict()
        torch.save(ckpt, path)
        # 同时落一份纯文本指标便于人读 | also drop human-readable metrics
        (self.out_dir / "last_metrics.json").write_text(
            json.dumps({"epoch": epoch, **metrics}, indent=2), encoding="utf-8"
        )
