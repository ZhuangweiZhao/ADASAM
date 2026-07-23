"""
SAM-RSP Stage 1: PSPNet 预训练 | PSPNet Pre-training on iSAID-5i Base Classes.
===============================================================================

在 iSAID-5i base classes 上训练 PSPNet (ResNet50 + PPM), 为 BAM 提供骨干权重。
Train PSPNet (ResNet50 + PPM) on iSAID-5i base classes to provide backbone
weights for the BAM (Rough Segment Prompt Generator).

用法 | Usage::

    python tools/sam_rsp_stage1.py --fold 0 --epochs 100 --data-root data/iSAID-5i

输出 | Output::

    runs/sam_rsp_stage1/fold0/best_model.pth  — PSPNet 权重 | best model weights
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Add SAM-RSP vendored code to path
_SAM_RSP = _REPO_ROOT / "thirdparty" / "SAM-RSP"
if str(_SAM_RSP) not in sys.path:
    sys.path.insert(0, str(_SAM_RSP))

from adasam.sam_rsp.dataset import PSPNetDataset, ISAID5I_FOLDS, IMAGENET_MEAN, IMAGENET_STD
from adasam.utils import set_seed


# ═══════════════════════════════════════════════════════════════════
# Transforms
# ═══════════════════════════════════════════════════════════════════

class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class Resize:
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, image, mask):
        import cv2
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        return image, mask


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if np.random.random() < self.p:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        return image, mask


class RandomScale:
    def __init__(self, scale_range=(0.5, 2.0)):
        self.scale_range = scale_range

    def __call__(self, image, mask):
        import cv2
        scale = np.random.uniform(*self.scale_range)
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return image, mask


class RandomCrop:
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, image, mask):
        import cv2
        h, w = image.shape[:2]
        th, tw = self.size
        if h < th or w < tw:
            image = cv2.resize(image, (max(w, tw), max(h, th)), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (max(w, tw), max(h, th)), interpolation=cv2.INTER_NEAREST)
            h, w = image.shape[:2]
        y = np.random.randint(0, h - th + 1)
        x = np.random.randint(0, w - tw + 1)
        return image[y:y+th, x:x+tw], mask[y:y+th, x:x+tw]


class Normalize:
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        image = (image - self.mean) / self.std
        return image, mask


class ToTensor:
    def __call__(self, image, mask):
        return image, mask


# ═══════════════════════════════════════════════════════════════════
# PSPNet Model Builder (torchvision ResNet50 + PPM)
# ═══════════════════════════════════════════════════════════════════

def build_pspnet(num_base_classes: int, pretrained: bool = True) -> nn.Module:
    """构建 PSPNet (torchvision ResNet50 + PPM + classifier).

    使用 torchvision 预训练 ResNet50, 应用 dilated convolutions, 添加 PPM + 分类头。
    Uses torchvision pretrained ResNet50 with dilated convolutions + PPM + classifier.
    与 SAM-RSP 原始 PSPNet 架构完全一致, 但避免了硬编码权重路径问题。
    """
    try:
        from torchvision.models import resnet50, ResNet50_Weights
    except ImportError:
        from torchvision.models import resnet50
        ResNet50_Weights = None

    if pretrained and ResNet50_Weights is not None:
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    else:
        resnet = resnet50(weights=None)

    # Extract layers (same as SAM-RSP PSPNet)
    layer0 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu,
        resnet.maxpool
    )
    layer1 = resnet.layer1
    layer2 = resnet.layer2
    layer3 = resnet.layer3
    layer4 = resnet.layer4

    # Apply dilation to layer3 and layer4 (maintain spatial resolution)
    for n, m in layer3.named_modules():
        if 'conv2' in n:
            m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
        elif 'downsample.0' in n:
            m.stride = (1, 1)
    for n, m in layer4.named_modules():
        if 'conv2' in n:
            m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
        elif 'downsample.0' in n:
            m.stride = (1, 1)

    encoder = nn.Sequential(layer0, layer1, layer2, layer3, layer4)
    fea_dim = 2048
    bins = (1, 2, 3, 6)

    # Import PPM from SAM-RSP
    sys.path.insert(0, str(_SAM_RSP))
    from model.PPM import PPM

    ppm = PPM(fea_dim, int(fea_dim / len(bins)), bins)
    cls_head = nn.Sequential(
        nn.Conv2d(fea_dim * 2, 512, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=0.1),
        nn.Conv2d(512, num_base_classes + 1, kernel_size=1),  # +1 for BG
    )

    class PSPNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.ppm = ppm
            self.cls = cls_head
            self.zoom_factor = 8
            self.classes = num_base_classes + 1
            self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        def forward(self, x, y=None):
            x_size = x.size()
            h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
            w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

            x = self.encoder(x)
            x = self.ppm(x)
            x = self.cls(x)

            if self.zoom_factor != 1:
                x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

            if self.training and y is not None:
                main_loss = self.criterion(x, y.long())
                return x.max(1)[1], main_loss
            return x

    return PSPNet()


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def train(args: argparse.Namespace) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    fold_def = ISAID5I_FOLDS[args.fold]
    base_classes = sorted(fold_def["train"])
    num_base = len(base_classes)
    print(f"[Stage1] fold={args.fold} base_classes={base_classes} num_base={num_base}")

    # Data
    train_transform = Compose([
        RandomScale((0.5, 2.0)),
        RandomCrop(args.crop_size),
        RandomHorizontalFlip(),
        Normalize(),
    ])
    val_transform = Compose([
        Normalize(),
    ])

    data_root = Path(args.data_root)
    train_ds = PSPNetDataset(data_root, fold=args.fold, transform=train_transform)
    val_ds = PSPNetDataset(data_root, fold=args.fold, transform=val_transform)

    # Use subset for validation (separate val split from train tiles)
    n_val = min(len(val_ds) // 10, 1000)  # 10% or max 1000
    val_indices = list(range(len(val_ds)))
    np.random.RandomState(args.seed).shuffle(val_indices)
    val_indices = val_indices[:n_val]
    from torch.utils.data import Subset
    val_subset = Subset(val_ds, val_indices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    print(f"[Stage1] train={len(train_ds)} val={len(val_subset)}")

    # Model
    model = build_pspnet(num_base, pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[Stage1] params={n_params:.1f}M trainable={n_trainable:.1f}M")

    # Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Output
    out_dir = Path(args.output_dir) / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_model.pth"
    best_miou = -1.0

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            _, loss = model(images, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        scheduler.step()
        train_loss /= max(len(train_loader), 1)

        # Validate
        model.eval()
        val_loss = 0.0
        inter_sum = np.zeros(num_base + 1)
        union_sum = np.zeros(num_base + 1)
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                logits = model(images, masks)  # eval mode: returns logits only
                loss = criterion(logits, masks.long())
                val_loss += loss.item()

                pred = logits.max(1)[1].cpu().numpy()
                target = masks.cpu().numpy()
                for c in range(num_base + 1):
                    inter_sum[c] += ((pred == c) & (target == c)).sum()
                    union_sum[c] += ((pred == c) | (target == c)).sum()

        val_loss /= max(len(val_loader), 1)
        iou_per_class = inter_sum[1:] / (union_sum[1:] + 1e-10)  # exclude BG
        miou = float(np.mean(iou_per_class))

        print(f"[{epoch:>3d}] train_loss={train_loss:.3f} val_loss={val_loss:.3f} "
              f"mIoU={miou:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        if miou > best_miou:
            best_miou = miou
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "miou": miou,
                "num_classes": num_base,
                "base_classes": base_classes,
                "fold": args.fold,
            }, best_path)
            print(f"  -> best: mIoU={miou:.4f}")

    print(f"\n[Stage1] Done. best_mIoU={best_miou:.4f} saved to {best_path}")
    return best_path


def main() -> None:
    p = argparse.ArgumentParser(description="SAM-RSP Stage 1: PSPNet Pre-training")
    p.add_argument("--fold", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--data-root", type=str, default=str(_REPO_ROOT / "data" / "iSAID-5i"))
    p.add_argument("--output-dir", type=str, default=str(_REPO_ROOT / "runs" / "sam_rsp_stage1"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=321)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
