"""
训练流程 | Training pipeline.
============================

AdaSAM 的**唯一**训练入口 (单一 Trainer 类, 无 if-mode 分支)。
The single training entry point of AdaSAM (one Trainer class, no if-mode branches).

范式 | Paradigm — teacher forcing:
    - 冻结 MobileSAM 图像编码器; 只训练 PromptMaskDecoder (PrototypeAdapter + MaskDecoder)。
      Frozen MobileSAM image encoder; only PromptMaskDecoder trains.
    - 每个 episode: 从 K 张 support 构建类原型; 在 query tile 上, 为每个 GT 实例在其内部
      采样一个提示点 (距离变换峰值) → 解码该实例掩码 → focal+dice 监督。
      Per episode: build a class prototype from K supports; on the query tile, sample one
      interior point per GT instance (distance-transform peak) → decode that instance → supervise.
    - 推理时提示点改由原型相似度峰值给出 (adasam.decoder.PromptMaskDecoder.forward), 与训练
      共用 decode() 核心。At inference, prompts come from prototype-sim peaks, sharing decode().
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from adasam.backbone import build_mobile_sam, MobileSAMBackbone
from adasam.datasets import EpisodeSampler, ISAIDInstanceDataset
from adasam.decoder import PromptMaskDecoder
from adasam.logging import get_logger
from adasam.logging.backends import ConsoleBackend, FileBackend
from adasam.losses import combined_loss, dice_loss, focal_loss, mask_iou
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

        # ── 模型装配 (一次构建 Sam, 分发到 backbone/decoder) | Assemble model (one Sam) ──
        ckpt = self._resolve(config["backbone"]["checkpoint"])
        sam = build_mobile_sam(ckpt, config["backbone"].get("model_type", "vit_t"), self.device)
        self.backbone = MobileSAMBackbone(sam.image_encoder, sam.image_encoder.img_size).to(self.device)
        self.image_size = self.backbone.img_size

        proto_cfg = config.get("prototype", {})
        self.embed_dim = int(proto_cfg.get("embed_dim", 256))
        self.decoder = PromptMaskDecoder(
            sam.prompt_encoder, sam.mask_decoder,
            embed_dim=self.embed_dim, image_size=self.image_size,
            top_k_points=int(proto_cfg.get("top_k_points", 10)),
            sim_threshold=float(proto_cfg.get("sim_threshold", 0.5)),
            min_distance=int(proto_cfg.get("min_distance", 1)),
            n_proto_tokens=int(proto_cfg.get("n_proto_tokens", 1)),
            train_mask_decoder=True, train_prompt_encoder=False,
        ).to(self.device)
        self.proto_builder = PrototypeBuilder(self.embed_dim)

        # ── 优化器 / 调度器 (仅可训练参数) | Optimizer / scheduler (trainable params only) ──
        tcfg = config["train"]
        self.epochs = int(tcfg.get("epochs", 50))
        self.episodes_per_epoch = int(tcfg.get("episodes_per_epoch", 200))
        self.grad_clip = float(tcfg.get("grad_clip", 1.0))
        self.max_instances = int(tcfg.get("max_instances_per_query", 32))
        self.iou_loss_weight = float(tcfg.get("iou_loss_weight", 1.0))
        params = [p for p in self.decoder.parameters() if p.requires_grad]
        self.optimizer = AdamW(params, lr=float(tcfg.get("lr", 1e-4)),
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
        n_train = sum(p.numel() for p in params) / 1e6
        self.logger.log_info("init",
                             f"device={self.device}, trainable={n_train:.2f}M, "
                             f"classes={self.sampler.eligible_classes()}, out={self.out_dir}")

    # ── 路径工具 | Path helper ──

    @staticmethod
    def _resolve(path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (_REPO_ROOT / p)

    # ── 嵌入 | Embedding ──

    def _embed(self, image: torch.Tensor) -> torch.Tensor:
        """tile 图像 [3,H,W]∈[0,1] → 图像嵌入 [1,256,64,64] (冻结, 无梯度) | frozen embedding."""
        x, _ = preprocess_image(image)
        return self.backbone(x.unsqueeze(0).to(self.device))["image_embedding"]

    # ── 原型构建 | Prototype build ──

    def _build_prototype(self, support_indices: list[int], class_id: int) -> torch.Tensor:
        """由 K 张 support 构建类原型 | Build class prototype from K supports."""
        embeddings, masks = [], []
        for idx in support_indices:
            sample = self.dataset[idx]
            fg = self._class_foreground(sample["instances"], class_id, self.tile_size)
            if fg is None:
                continue
            embeddings.append(self._embed(sample["image"])[0])   # [256,64,64]
            masks.append(fg)
        return self.proto_builder.build(embeddings, masks)

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

    # ── GT 教师提示点 + 掩码 | Teacher-forced GT points + masks ──

    def _query_targets(
        self, query_sample: dict, class_id: int
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """query tile 内每个类实例 → (输入帧提示点, labels, GT 掩码) | per-instance points + masks."""
        insts = [i for i in query_sample["instances"] if i["category_id"] == class_id]
        if not insts:
            return None
        if len(insts) > self.max_instances:
            insts = self._rng.sample(insts, self.max_instances)

        coords_tile, gt_masks = [], []
        for inst in insts:
            xy = self._interior_point(inst["mask"])
            if xy is None:
                continue
            coords_tile.append(xy)
            gt_masks.append(inst["mask"].float())
        if not coords_tile:
            return None

        coords_tile_t = torch.tensor(coords_tile, dtype=torch.float32, device=self.device)  # [N,2]
        coords_in = PromptMaskDecoder.scale_points(
            coords_tile_t, (self.tile_size, self.tile_size), (self.image_size, self.image_size)
        )
        labels = torch.ones(coords_in.shape[0], dtype=torch.float32, device=self.device)
        gt = torch.stack(gt_masks, dim=0).to(self.device)               # [N,H,W]
        return coords_in, labels, gt

    @staticmethod
    def _interior_point(mask: torch.Tensor) -> Optional[tuple[float, float]]:
        """实例内部最深点 (距离变换峰值), tile 帧 (x,y) | deepest interior point (distance transform)."""
        m = mask.cpu().numpy().astype(np.uint8)
        if m.sum() == 0:
            return None
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        flat = int(dt.argmax())
        y, x = divmod(flat, m.shape[1])
        return float(x), float(y)

    # ── 单 episode 训练 | Single-episode training step ──

    def _train_episode(self, episode: dict) -> Optional[dict]:
        cls = episode["class_id"]
        prototype = self._build_prototype(episode["support_indices"], cls)

        query = self.dataset[episode["query_index"]]
        targets = self._query_targets(query, cls)
        if targets is None:
            return None
        coords, labels, gt = targets                        # [N,2],[N],[N,H,W]

        emb = self._embed(query["image"])                   # [1,256,64,64]
        low_res, iou_pred = self.decoder.decode(emb, prototype, coords, labels)
        logits = self.decoder.upscale_logits(
            low_res, (self.image_size, self.image_size), (self.tile_size, self.tile_size)
        )                                                    # [N,H,W]

        fl = focal_loss(logits, gt)
        dl = dice_loss(logits, gt)
        iou_target = mask_iou(torch.sigmoid(logits), gt)     # [N] detached
        iou_head_loss = torch.nn.functional.mse_loss(iou_pred[:, 0], iou_target)
        loss = fl + dl + self.iou_loss_weight * iou_head_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.decoder.parameters() if p.requires_grad], self.grad_clip
        )
        self.optimizer.step()

        return {
            "loss": float(loss.detach()), "focal": float(fl.detach()),
            "dice": float(dl.detach()), "iou_head": float(iou_head_loss.detach()),
            "pred_iou": float(iou_target.mean()), "n_inst": gt.shape[0],
        }

    # ── 主循环 | Main loop ──

    def train(self) -> Path:
        """运行训练, 返回最优 checkpoint 路径 | Run training, return best checkpoint path."""
        self.decoder.train()
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
            self.scheduler.step()

            mean = {k: v / max(n, 1) for k, v in agg.items()}
            for k, v in mean.items():
                self.logger.log_metric(f"train/{k}", v, step=epoch, phase="train")
            self.logger.log_info("epoch",
                                 f"epoch {epoch}: loss={mean.get('loss', 0):.4f} "
                                 f"dice={mean.get('dice', 0):.4f} n={n}", step=epoch)

            self._save(self.out_dir / "last_model.pt", epoch, mean)
            if mean.get("loss", float("inf")) < best_loss:
                best_loss = mean["loss"]
                self._save(best_path, epoch, mean)

        self.logger.flush()
        return best_path

    # ── Checkpoint ──

    def _save(self, path: Path, epoch: int, metrics: dict) -> None:
        """保存 checkpoint (统一 schema, 无条件键) | Save checkpoint (uniform schema)."""
        torch.save({
            "epoch": epoch,
            "model": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.cfg,
            "metrics": metrics,
        }, path)
        # 同时落一份纯文本指标便于人读 | also drop human-readable metrics
        (self.out_dir / "last_metrics.json").write_text(
            json.dumps({"epoch": epoch, **metrics}, indent=2), encoding="utf-8"
        )
