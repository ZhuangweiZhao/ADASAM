"""
NEU_Seg 多类别分割训练 (FastSAM-aligned Edition) | Multi-class Segmentation Training.
======================================================================================

基于 AdaTile-FastSAM 已验证的最佳实践重写:
    - 无 prototype conditioning (PureDecoder 风格, ablation 证明 prototype 无用)
    - 无 1024 preprocess (pad to 32, 原生分辨率)
    - Focal Loss γ=5.0 + Dice Loss (极致类别不平衡)
    - InstanceNorm (小 batch 友好)
    - P3+P4 双尺度 CNN decoder
    - 丰富数据增强

用法 | Usage::
    python tools/train_neuseg.py --device cuda --epochs 200
    python tools/train_neuseg.py --device cuda --epochs 200 --k-shot 5
"""

from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adasam.backbone import MultiScaleMobileSAMBackbone
from adasam.datasets import NEUSegDataset
from adasam.logging import get_logger
from adasam.logging.backends import ConsoleBackend, FileBackend
from adasam.utils import set_seed


# ═══════════════════════════════════════════════════════════════════
# Decoder: Pure P3P4 (no prototype, FastSAM-aligned)
# ═══════════════════════════════════════════════════════════════════

class PureDecoderP3P4(nn.Module):
    """双尺度纯 CNN 解码器 (无 prototype) | Two-scale pure CNN decoder (no prototype).

    FastSAM PureDecoderP3P4 的 MobileSAM 等效实现。
    P3 (H/8): stage1 features [128, H/8, W/8] → 边界/细节
    P4 (H/16): neck features [256, H/16, W/16] → 语义/主力

    架构: project → refine → upsample → concat → fuse → H/4 output.
    """

    def __init__(self, p3_ch: int = 128, p4_ch: int = 256, out_ch: int = 4,
                 mid_ch: int = 128):
        super().__init__()
        self.out_channels = out_ch

        # P4 path (main, H/16)
        self.p4_proj = nn.Sequential(
            nn.Conv2d(p4_ch, mid_ch, 1, bias=False),
            nn.InstanceNorm2d(mid_ch, affine=True), nn.ReLU(inplace=True))
        self.p4_refine = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch // 2, 3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_ch // 2, affine=True), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch // 2, mid_ch // 4, 3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_ch // 4, affine=True), nn.ReLU(inplace=True))
        self.p4_head = nn.Sequential(
            nn.Conv2d(mid_ch // 4, 32, 3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True), nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, 1))

        # P3 path (boundary, H/8)
        self.p3_proj = nn.Sequential(
            nn.Conv2d(p3_ch, mid_ch // 2, 1, bias=False),
            nn.InstanceNorm2d(mid_ch // 2, affine=True), nn.ReLU(inplace=True))
        self.p3_refine = nn.Sequential(
            nn.Conv2d(mid_ch // 2, mid_ch // 4, 3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_ch // 4, affine=True), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch // 4, mid_ch // 4, 3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_ch // 4, affine=True), nn.ReLU(inplace=True))
        self.p3_head = nn.Sequential(
            nn.Conv2d(mid_ch // 4, 32, 3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True), nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, 1))

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_ch * 2, 16, 3, padding=1, bias=False),
            nn.InstanceNorm2d(16, affine=True), nn.ReLU(inplace=True),
            nn.Conv2d(16, out_ch, 1))

        self._init_weights()
        self._n_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None: nn.init.ones_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor:
        """Return: [B, out_ch, H/4, W/4] raw logits."""
        # P4: H/16 → H/4
        p4_x = self.p4_proj(p4)
        p4_x = self.p4_refine(p4_x)
        p4_logit = self.p4_head(p4_x)

        # P3: H/8 → H/4
        p3_x = self.p3_proj(p3)
        p3_x = self.p3_refine(p3_x)
        p3_logit = self.p3_head(p3_x)

        H_out, W_out = p3_logit.shape[2] * 2, p3_logit.shape[3] * 2
        p4_up = F.interpolate(p4_logit, size=(H_out, W_out), mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3_logit, size=(H_out, W_out), mode='bilinear', align_corners=False)
        fused = torch.cat([p3_up, p4_up], dim=1)
        return self.fusion(fused)


# ═══════════════════════════════════════════════════════════════════
# Loss: Focal (γ=5.0) + Dice (FastSAM-aligned)
# ═══════════════════════════════════════════════════════════════════

def focal_loss(logits: torch.Tensor, target: torch.Tensor,
               gamma: float = 5.0, ignore_index: int = 255) -> torch.Tensor:
    """Focal Loss γ=5.0 for extreme class imbalance."""
    ce = F.cross_entropy(logits, target, ignore_index=ignore_index, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def dice_loss(probs: torch.Tensor, target: torch.Tensor,
              smooth: float = 1e-8) -> torch.Tensor:
    """Per-FG-class Dice, ignore BG and absent classes."""
    C = probs.shape[1]
    total = 0.0; valid = 0
    for c in range(1, C):
        p_c = probs[:, c]
        t_c = (target == c).float()
        if t_c.sum() == 0: continue
        inter = (p_c * t_c).sum()
        union = p_c.sum() + t_c.sum()
        total += (2 * inter + smooth) / (union + smooth)
        valid += 1
    return 1.0 - total / max(valid, 1) if valid > 0 else torch.tensor(0.0, device=probs.device)


def combined_loss(logits: torch.Tensor, target: torch.Tensor,
                  alpha: float = 0.5, gamma: float = 5.0
                  ) -> tuple[torch.Tensor, dict]:
    probs = F.softmax(logits, dim=1)
    fl = focal_loss(logits, target, gamma=gamma)
    dl = dice_loss(probs, target)
    return alpha * fl + (1 - alpha) * dl, {"focal": fl.item(), "dice": dl.item()}


# ═══════════════════════════════════════════════════════════════════
# Data Augmentation (FastSAM-aligned)
# ═══════════════════════════════════════════════════════════════════

class SegAug:
    def __init__(self, p_flip=0.5, p_rotate=0.5, brightness=0.2, contrast=0.2,
                 noise_std=0.02):
        self.p_flip = p_flip; self.p_rotate = p_rotate
        self.brightness = brightness; self.contrast = contrast
        self.noise_std = noise_std

    def __call__(self, img: torch.Tensor, mask: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.p_flip:
            img = torch.flip(img, [-1]); mask = torch.flip(mask, [-1])
        if random.random() < self.p_flip:
            img = torch.flip(img, [-2]); mask = torch.flip(mask, [-2])
        if random.random() < self.p_rotate:
            k = random.randint(0, 3)
            img = torch.rot90(img, k, [-2, -1]); mask = torch.rot90(mask, k, [-2, -1])
        b = random.uniform(-self.brightness, self.brightness)
        c = random.uniform(1 - self.contrast, 1 + self.contrast)
        img = torch.clamp(img * c + b, 0, 1)
        if self.noise_std > 0:
            img = torch.clamp(img + torch.randn_like(img) * self.noise_std, 0, 1)
        return img, mask


# ═══════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════

def pad_to_32(t, mask=None):
    if t.dim() == 4: H, W = t.shape[2], t.shape[3]
    else: H, W = t.shape[1], t.shape[2]
    ph, pw = (32 - H % 32) % 32, (32 - W % 32) % 32
    if ph == 0 and pw == 0: return t, mask, (H, W)
    tp = F.pad(t, (0, pw, 0, ph), value=0)
    mp = F.pad(mask, (0, pw, 0, ph), value=0) if mask is not None else None
    return tp, mp, (H, W)


@torch.no_grad()
def evaluate(decoder, backbone, dataset, device, num_classes=4, max_samples=0):
    decoder.eval()
    per_class_inter = torch.zeros(num_classes, device=device)
    per_class_union = torch.zeros(num_classes, device=device)
    correct = 0; total = 0

    indices = list(range(len(dataset)))
    if max_samples > 0: indices = indices[:max_samples]

    for idx in tqdm(indices, desc="Eval", leave=False):
        s = dataset[idx]
        img = s["image"]; gt = s["masks"].squeeze(0).long().to(device)
        H, W = gt.shape

        img_p, _, _ = pad_to_32(img.unsqueeze(0))
        feats = backbone(img_p.to(device))
        logits = decoder(feats["stage1"], feats["stage3"])
        pred = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        pred_c = pred.argmax(1).squeeze(0)

        correct += (pred_c == gt).sum().item(); total += gt.numel()
        for c in range(num_classes):
            pc = (pred_c == c); gc = (gt == c)
            per_class_inter[c] += (pc & gc).sum()
            per_class_union[c] += (pc | gc).sum()

    ious = {}
    valid = []
    for c in range(num_classes):
        i = per_class_inter[c].item(); u = per_class_union[c].item()
        iou = i / u if u > 0 else float("nan")
        ious[NEUSegDataset.CLASS_NAMES[c]] = round(iou, 6)
        if iou == iou: valid.append(iou)
    return {
        "mIoU": round(float(np.mean(valid)), 6) if valid else 0.0,
        "pixel_accuracy": round(correct / total, 6) if total > 0 else 0.0,
        "per_class_IoU": ious, "n_evaluated": len(indices),
    }


# ═══════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════

class NeuSegTrainer:
    def __init__(self, cfg: dict, args: argparse.Namespace):
        self.cfg = cfg; self.args = args
        self.seed = int(cfg.get("seed", 42))
        set_seed(self.seed)
        self.device = torch.device(cfg["train"].get("device", "cuda")
                                   if torch.cuda.is_available() else "cpu")
        self._rng = random.Random(self.seed)

        self.k_shot = int(cfg["fewshot"].get("k_shot", 3))
        self.steps_per_epoch = int(cfg["fewshot"].get("steps_per_epoch", 200))

        # Data
        dr = self._resolve(cfg["data"]["data_root"])
        self.train_ds = NEUSegDataset(root=dr, split="train")
        self.val_ds = NEUSegDataset(root=dr, split="test")
        self.num_classes = cfg["data"].get("num_classes", 4)

        # Augmentation
        aug_cfg = cfg.get("augmentation", {})
        self.aug = SegAug(
            p_flip=float(aug_cfg.get("p_flip", 0.5)),
            p_rotate=float(aug_cfg.get("p_rotate", 0.5)),
            brightness=float(aug_cfg.get("brightness", 0.2)),
            contrast=float(aug_cfg.get("contrast", 0.2)),
            noise_std=float(aug_cfg.get("noise_std", 0.02)),
        ) if aug_cfg.get("enabled", True) else None

        # Model: backbone (frozen) + pure decoder
        ckpt_bb = self._resolve(cfg["backbone"]["checkpoint"])
        target_sz = int(cfg["backbone"].get("img_size", 224))
        self.backbone = MultiScaleMobileSAMBackbone.build(
            str(ckpt_bb), cfg["backbone"].get("model_type", "vit_t"),
            self.device, img_size=target_sz,
        )
        self.backbone.eval()

        d_cfg = cfg.get("decoder", {})
        self.decoder = PureDecoderP3P4(
            p3_ch=int(d_cfg.get("p3_channels", 128)),
            p4_ch=int(d_cfg.get("p4_channels", 256)),
            out_ch=self.num_classes,
            mid_ch=int(d_cfg.get("mid_channels", 128)),
        ).to(self.device)

        # Optimizer
        tcfg = cfg["train"]
        self.epochs = int(tcfg.get("epochs", 200))
        self.optimizer = AdamW(self.decoder.parameters(),
                               lr=float(tcfg.get("lr", 1e-4)),
                               weight_decay=float(tcfg.get("weight_decay", 1e-4)))
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.grad_clip = float(tcfg.get("grad_clip", 1.0))
        self.eval_every = int(tcfg.get("eval_every", 5))

        # Output
        exp = f"neuseg_p3p4_k{self.k_shot}_s{self.seed}"
        self.out_dir = self._resolve(cfg.get("output_dir", "runs")) / exp
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("trainer.neuseg")
        if not self.logger.backends:
            self.logger.add_backend(ConsoleBackend())
            self.logger.add_backend(FileBackend(str(self.out_dir / "train.jsonl")))

        n_dec = sum(p.numel() for p in self.decoder.parameters()) / 1e3
        n_bb = sum(p.numel() for p in self.backbone.parameters()) / 1e6
        self.logger.log_info("init",
            f"device={self.device} decoder={n_dec:.1f}K backbone={n_bb:.1f}M(frozen) "
            f"k={self.k_shot} train={len(self.train_ds)} val={len(self.val_ds)} "
            f"aug={self.aug is not None} out={self.out_dir}")

    @staticmethod
    def _resolve(p): p = Path(p); return p if p.is_absolute() else _REPO_ROOT / p

    def train(self) -> Path:
        best_miou = 0.0; best_path = self.out_dir / "best_model.pt"
        all_idx = list(range(len(self.train_ds)))

        for epoch in range(1, self.epochs + 1):
            self.decoder.train()
            losses = []
            pbar = tqdm(range(self.steps_per_epoch), desc=f"E{epoch:3d}/{self.epochs}")

            for _ in pbar:
                q_idx = self._rng.randint(0, len(self.train_ds) - 1)
                q = self.train_ds[q_idx]
                img = q["image"]; gt = q["masks"].squeeze(0).long().to(self.device)

                if (gt > 0).sum() < 1: continue

                if self.aug is not None:
                    img, gt_t = self.aug(img, gt.unsqueeze(0))
                    gt = gt_t.squeeze(0).long()

                H, W = gt.shape
                img_p, _, _ = pad_to_32(img.unsqueeze(0))
                feats = self.backbone(img_p.to(self.device))
                logits = self.decoder(feats["stage1"], feats["stage3"])
                # Resize logits to GT size
                logits_up = F.interpolate(logits, size=(H, W), mode='bilinear',
                                          align_corners=False)

                loss, ld = combined_loss(logits_up, gt.unsqueeze(0),
                                         alpha=float(self.cfg["loss"].get("ce_weight", 0.5)),
                                         gamma=float(self.cfg["loss"].get("focal_gamma", 5.0)))

                if torch.isnan(loss) or torch.isinf(loss): continue

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.decoder.parameters(), self.grad_clip)
                self.optimizer.step()
                losses.append(loss.item())
                pbar.set_postfix(loss=f"{np.mean(losses[-50:]):.4f}" if losses else "?")

            self.scheduler.step()
            avg = np.mean(losses) if losses else 0.0
            self.logger.log_info("epoch", f"Epoch {epoch:3d} | loss={avg:.4f}")
            self.logger.log_metric("loss", avg, step=epoch, tags=["neuseg_train"])

            if epoch % self.eval_every == 0 or epoch == self.epochs:
                r = evaluate(self.decoder, self.backbone, self.val_ds, self.device,
                             num_classes=self.num_classes)
                miou = r["mIoU"]
                self.logger.log_info("eval",
                    f"  mIoU={miou:.4f} PA={r['pixel_accuracy']:.4f} best={best_miou:.4f}")
                for cn, iou_c in r["per_class_IoU"].items():
                    self.logger.log_info("eval", f"    {cn}: IoU={iou_c:.4f}")
                self.logger.log_metric("mIoU", miou, step=epoch, tags=["neuseg_eval"])

                if miou > best_miou:
                    best_miou = miou; self._save(best_path, epoch, miou)

        self._save(self.out_dir / "last_model.pt", self.epochs, best_miou)
        return best_path

    def _save(self, path, epoch, miou):
        torch.save({
            "epoch": epoch, "mIoU": miou, "num_classes": self.num_classes,
            "class_names": self.train_ds.CLASS_NAMES,
            "decoder_state_dict": {k: v.clone() for k, v in self.decoder.state_dict().items()},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.cfg, "args": vars(self.args),
        }, path)
        (self.out_dir / "last_metrics.json").write_text(
            json.dumps({"epoch": epoch, "mIoU": miou}, indent=2), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="AdaSAM NEU_Seg FastSAM-aligned Training")
    p.add_argument("--config", default=str(_REPO_ROOT / "configs" / "neu_seg.yaml"))
    p.add_argument("--k-shot", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--steps-per-epoch", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--no-aug", action="store_true", default=None)
    return p.parse_args()


def load_config(args):
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for keys, val in [
        (("fewshot","k_shot"), args.k_shot), (("train","epochs"), args.epochs),
        (("fewshot","steps_per_epoch"), args.steps_per_epoch), (("train","lr"), args.lr),
        (("train","device"), args.device), (("seed",), args.seed),
        (("output_dir",), args.output_dir), (("data","data_root"), args.data_root),
    ]:
        if val is not None:
            t = cfg; [t := t.setdefault(k, {}) for k in keys[:-1]]; t[keys[-1]] = val
    if args.no_aug is not None: cfg.setdefault("augmentation", {})["enabled"] = False
    return cfg


def main():
    args = parse_args(); cfg = load_config(args)
    best = NeuSegTrainer(cfg, args).train()
    print(f"\n[train_neuseg] done. best: {best}")


if __name__ == "__main__":
    main()
