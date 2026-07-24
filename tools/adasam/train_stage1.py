"""
AdaSAM Stage 1 — 领域适配训练 | Domain Adaptation Training.
=============================================================

Stage 1: MobileSAM (frozen) + CATAdapter + SegHead → 标准语义分割.
目标: Domain-aware Initialization — 让 MobileSAM 学会 iSAID 遥感特征,
为 Stage 2 的 Few-shot Semantic Learning 提供领域适应后的特征初始化。

MobileSAM (frozen) + CATAdapter + SegHead (1x1 Conv) → standard
semantic segmentation on base classes. Goal: domain-aware feature
initialization for Stage 2 few-shot learning.

用法 | Usage::

    # 完整训练
    python tools/adasam/train_stage1.py --fold 0 --epochs 50

    # 冒烟测试
    python tools/adasam/train_stage1.py --fold 0 --epochs 1 --steps 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adasam.adapters import CATAdapter
from adasam.backbone import build_mobile_sam, MobileSAMBackbone
from adasam.datasets.isaid_5i import ISAID5iDataset, ISAID5I_CATEGORIES
from adasam.losses.seg_losses import dice_loss, focal_loss
from adasam.utils import set_seed
from adasam.utils.transforms import preprocess_image


# ═══════════════════════════════════════════════════════════════════
# SegHead: 极简线性探针 | minimal linear probe
# ═══════════════════════════════════════════════════════════════════

class SegHead(nn.Module):
    """1×1 Conv → logits [B, num_classes, H, W] (上采样到 tile 分辨率)."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Conv2d(in_dim, num_classes, kernel_size=1)
        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        """x [B, C, gh, gw] → [B, num_classes, H, W]."""
        logits = self.head(x)
        return F.interpolate(logits, target_size, mode="bilinear", align_corners=False)


# ═══════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════

class Stage1Trainer:
    """Stage 1 领域适配训练器 | Domain adaptation trainer."""

    def __init__(self, cfg: dict, args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.args = args
        self.seed = int(cfg.get("seed", 42))
        set_seed(self.seed)
        self.device = torch.device(
            cfg["train"].get("device", "cuda") if torch.cuda.is_available() else "cpu"
        )

        # ── Data ──
        self.fold = int(cfg["data"].get("fold", 0))
        data_root = Path(cfg["data"]["data_root"])
        if not data_root.is_absolute():
            data_root = _REPO_ROOT / data_root

        tcfg = cfg["train"]
        self.epochs = int(tcfg.get("epochs", 50))
        self.batch_size = int(tcfg.get("batch_size", 8))
        self.val_every = int(tcfg.get("val_every", 5))

        # Base classes (stage 1 uses mode="base" — only base classes visible)
        self.train_ds = ISAID5iDataset(
            root=str(data_root), fold=self.fold, split="train", mode="base"
        )
        self.val_ds = ISAID5iDataset(
            root=str(data_root), fold=self.fold, split="val", mode="base"
        )
        self.num_base_classes = len(self.train_ds.visible_classes())
        print(f"[Stage1] fold={self.fold} base_classes={self.train_ds.visible_classes()} "
              f"train={len(self.train_ds)} val={len(self.val_ds)}")

        # ── Model ──
        ckpt_path = _REPO_ROOT / cfg["backbone"]["checkpoint"]
        sam = build_mobile_sam(str(ckpt_path), cfg["backbone"].get("model_type", "vit_t"), self.device)
        self.backbone = MobileSAMBackbone(sam.image_encoder, sam.image_encoder.img_size).to(self.device)
        self.embed_dim = 256

        # CATAdapter
        adapter_cfg = cfg.get("adapter", {})
        self.adapter = CATAdapter(
            dim=self.embed_dim,
            bottleneck=int(adapter_cfg.get("bottleneck", 64)),
        ).to(self.device)

        # SegHead — 1×1 Conv, linear probe
        self.seg_head = SegHead(
            in_dim=self.embed_dim,
            num_classes=self.num_base_classes,
        ).to(self.device)

        # Trainable params: adapter + seg_head only
        self._trainable = list(self.adapter.parameters()) + list(self.seg_head.parameters())
        n_train = sum(p.numel() for p in self._trainable) / 1e6
        print(f"[Stage1] trainable params: {n_train:.2f}M")

        # ── Optimizer ──
        lr = float(tcfg.get("lr", 1e-3))
        self.optimizer = AdamW(self._trainable, lr=lr, weight_decay=float(tcfg.get("weight_decay", 1e-4)))
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        # ── Output ──
        exp = f"stage1_fold{self.fold}_seed{self.seed}"
        self.out_dir = _REPO_ROOT / cfg.get("output_dir", "runs") / exp
        self.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Stage1] output: {self.out_dir}")

        # ── Loss weights ──
        loss_cfg = cfg.get("loss", {})
        self.focal_weight = float(loss_cfg.get("focal_weight", 1.0))
        self.dice_weight = float(loss_cfg.get("dice_weight", 1.0))
        self.focal_gamma = float(loss_cfg.get("focal_gamma", 5.0))
        self.focal_eps = float(loss_cfg.get("focal_eps", 1e-4))

    # ── Validation ──

    def _validate(self, class_to_idx: dict) -> dict:
        """Compute mIoU + accuracy on validation set.

        :param class_to_idx: {class_id → contiguous_label (0..N-1)}.
        :return: {"mIoU": float, "acc": float, "per_class_iou": dict}.
        """
        self.backbone.eval()
        self.adapter.eval()
        self.seg_head.eval()

        idx_to_class = {v: k for k, v in class_to_idx.items()}
        num_cls = len(class_to_idx)
        inter = torch.zeros(num_cls, dtype=torch.float64)
        union = torch.zeros(num_cls, dtype=torch.float64)
        correct = 0
        total = 0

        for idx in range(len(self.val_ds)):
            sample = self.val_ds[idx]
            x, _ = preprocess_image(sample["image"])
            x = x.unsqueeze(0).to(self.device)

            # Ground truth (255 = ignore/background, consistent with CE ignore_index)
            gt = torch.full((256, 256), 255, dtype=torch.long)
            for cls_id in self.val_ds.visible_classes():
                mask = self.val_ds.get_class_mask(idx, cls_id)
                if mask is not None and mask.sum() > 0:
                    gt[mask > 0.5] = class_to_idx[cls_id]

            with torch.no_grad():
                emb = self.backbone(x)["image_embedding"]
                adapted = self.adapter(emb)
                logits = self.seg_head(adapted, (256, 256))
                pred = logits[0].argmax(dim=0).cpu()

            # mIoU (per-class)
            for c in range(num_cls):
                pred_c = pred == c
                gt_c = gt == c
                inter[c] += (pred_c & gt_c).sum().item()
                union[c] += (pred_c | gt_c).sum().item()

            # FG pixel accuracy (255 = ignore)
            fg = gt != 255
            correct += (pred[fg] == gt[fg]).sum().item()
            total += fg.sum().item()

        per_class_iou = {}
        for c in range(num_cls):
            cls_id = idx_to_class[c]
            iou = (inter[c] / union[c]).item() if union[c] > 0 else float("nan")
            per_class_iou[str(cls_id)] = round(iou, 4)

        valid_ious = [v for v in per_class_iou.values() if not (v != v)]  # filter nan
        miou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
        acc = correct / max(total, 1)

        self.adapter.train()
        self.seg_head.train()

        return {"miou": round(miou, 4), "acc": round(acc, 4), "per_class_iou": per_class_iou}

    # ── Training ──

    def _build_gt(self, index: int, class_to_idx: dict) -> torch.Tensor:
        """Build multiclass GT label map for a single tile.

        :param index: dataset index.
        :param class_to_idx: {class_id → contiguous_label}.
        :return: [256, 256] long tensor, 255 = ignore/background.
        """
        gt = torch.full((256, 256), 255, dtype=torch.long)
        for cls_id in self.train_ds.visible_classes():
            mask = self.train_ds.get_class_mask(index, cls_id)
            if mask is not None and mask.sum() > 0:
                gt[mask > 0.5] = class_to_idx[cls_id]
        return gt

    def train(self) -> Path:
        """Run full training loop with manual batching."""
        class_to_idx = {c: i for i, c in enumerate(sorted(self.train_ds.visible_classes()))}
        all_indices = list(range(len(self.train_ds)))
        rng = random.Random(self.seed)

        best_path = self.out_dir / "best_adapter.pt"
        best_miou = -1.0
        max_steps = getattr(self.args, "steps", None)

        for epoch in range(self.epochs):
            self.backbone.eval()
            self.adapter.train()
            self.seg_head.train()

            rng.shuffle(all_indices)
            total_loss = 0.0
            n_batches = 0

            pbar = tqdm(range(0, len(all_indices), self.batch_size), desc=f"epoch {epoch}")
            for start in pbar:
                batch_indices = all_indices[start:start + self.batch_size]
                if len(batch_indices) < 2:
                    continue

                # Load batch: preprocess images → backbone → adapter
                images_1024 = []
                gts = []
                for idx in batch_indices:
                    sample = self.train_ds[idx]
                    x, _ = preprocess_image(sample["image"])
                    images_1024.append(x)
                    gts.append(self._build_gt(idx, class_to_idx))

                x = torch.stack(images_1024, dim=0).to(self.device)
                gt_batch = torch.stack(gts, dim=0).to(self.device)

                with torch.no_grad():
                    emb = self.backbone(x)["image_embedding"]
                adapted = self.adapter(emb)
                logits = self.seg_head(adapted, (256, 256))

                # Loss: CE + Focal + Dice
                ce = F.cross_entropy(logits, gt_batch, ignore_index=255)
                prob = logits.softmax(dim=1)
                fg_mask = gt_batch != 255  # all foreground classes (including index 0)
                focal = focal_loss(
                    prob.max(dim=1)[0], fg_mask.float(),
                    gamma=self.focal_gamma, eps=self.focal_eps,
                )
                dice = dice_loss(prob.max(dim=1)[0], fg_mask.float())

                loss = ce + self.focal_weight * focal + self.dice_weight * dice

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.3f}")

                if max_steps and n_batches >= max_steps:
                    break

            self.scheduler.step()
            avg_loss = total_loss / max(n_batches, 1)
            parts = [f"loss={avg_loss:.4f}", f"lr={self.optimizer.param_groups[0]['lr']:.2e}"]

            # Validate every epoch
            metrics = self._validate(class_to_idx)
            is_best = metrics["miou"] > best_miou
            tag = " ★" if is_best else ""
            parts.append(f"val_mIoU={metrics['miou']:.4f} val_acc={metrics['acc']:.4f}{tag}")
            # Per-class IoU for diagnosis
            pcio = metrics.get("per_class_iou", {})
            pcio_str = " ".join(f"c{k}={v:.3f}" for k, v in sorted(pcio.items(), key=lambda x: int(x[0])))
            parts.append(f"per_cls=[{pcio_str}]")

            if is_best:
                best_miou = metrics["miou"]
                self._save(best_path, epoch, avg_loss, metrics)

            print(f"[Stage1] epoch {epoch:>3d}: " + " | ".join(parts))

            if max_steps and n_batches >= max_steps:
                break

        print(f"[Stage1] done. best_mIoU={best_miou:.4f} → {best_path}")
        return best_path

    def _save(self, path: Path, epoch: int, loss: float, metrics: dict | None = None) -> None:
        data: dict = {
            "epoch": epoch,
            "stage": "stage1",
            "adapter": self.adapter.state_dict(),
            "fold": self.fold,
            "num_base_classes": self.num_base_classes,
            "loss": loss,
            "config": self.cfg,
        }
        if metrics:
            data["metrics"] = metrics
        torch.save(data, path)


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AdaSAM Stage 1: Domain Adaptation")
    p.add_argument("--config", default=str(_REPO_ROOT / "configs" / "stage1.yaml"))
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--steps", type=int, default=None, help="limit steps per epoch (smoke test)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--weights", default=None, help="MobileSAM weights path override")
    return p.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    overrides = [
        (("data", "fold"), args.fold),
        (("train", "epochs"), args.epochs),
        (("train", "batch_size"), args.batch_size),
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
                d = d.setdefault(k, {})
            d[keys[-1]] = val
    if args.weights is not None:
        cfg.setdefault("backbone", {})["checkpoint"] = args.weights
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    trainer = Stage1Trainer(cfg, args)
    trainer.train()


if __name__ == "__main__":
    main()
