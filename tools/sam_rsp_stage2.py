"""
SAM-RSP Stage 2: BAM (RSPG) Meta-Learner Training on iSAID-5i Base Classes.
=============================================================================

在 Stage 1 PSPNet 权重基础上训练 BAM 元学习器 (Rough Segment Prompt Generator)。
Train the BAM meta-learner with frozen PSPNet backbone + base learner on
few-shot FG/BG episodes (base classes only).

用法 | Usage::

    python tools/sam_rsp_stage2.py --fold 0 --shot 1 --epochs 50 \\
        --stage1-ckpt runs/sam_rsp_stage1/fold0/best_model.pth

输出 | Output::

    runs/sam_rsp_stage2/fold0_shot1/best_model.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adasam.sam_rsp.bam import BAMModel
from adasam.sam_rsp.dataset import FewShotEpisodeDataset, ISAID5I_FOLDS, IMAGENET_MEAN, IMAGENET_STD
from adasam.utils import set_seed


# ═══════════════════════════════════════════════════════════════════
# Transforms (same design as Stage 1)
# ═══════════════════════════════════════════════════════════════════

class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
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
        return image[y:y + th, x:x + tw], mask[y:y + th, x:x + tw]


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if np.random.random() < self.p:
            image = np.ascontiguousarray(np.fliplr(image))
            mask = np.ascontiguousarray(np.fliplr(mask))
        return image, mask


class Resize:
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, image, mask):
        import cv2
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        return image, mask


class Normalize:
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        image = (image - self.mean) / self.std
        return image, mask


# ═══════════════════════════════════════════════════════════════════
# Dataset wrapper: applies transforms to query + all support pairs
# ═══════════════════════════════════════════════════════════════════

class EpisodeTransformWrapper:
    """Apply transform to query and each support image/mask independently."""

    def __init__(self, base_dataset: FewShotEpisodeDataset, transform):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        q_image = item["query_image"].numpy().transpose(1, 2, 0)  # [3,H,W] → [H,W,3]
        q_mask = item["query_mask"].numpy()                        # [H,W]

        # Apply transform to query
        q_image, q_mask = self.transform(q_image, q_mask)

        # Apply transform to each support
        s_images = item["support_images"].numpy()  # [K, 3, H, W]
        s_masks = item["support_masks"].numpy()    # [K, H, W]
        K = s_images.shape[0]
        s_imgs_out, s_msks_out = [], []
        for k in range(K):
            s_img_k = s_images[k].transpose(1, 2, 0)  # [3,H,W] → [H,W,3]
            s_msk_k = s_masks[k]
            si, sm = self.transform(s_img_k, s_msk_k)
            s_imgs_out.append(torch.from_numpy(si).permute(2, 0, 1))  # [3,H,W]
            s_msks_out.append(torch.from_numpy(sm))

        return {
            "query_image": torch.from_numpy(q_image).permute(2, 0, 1).float(),  # [3,H,W]
            "query_mask": torch.from_numpy(q_mask).long(),                       # [H,W]
            "support_images": torch.stack(s_imgs_out, 0).float(),                # [K,3,H,W]
            "support_masks": torch.stack(s_msks_out, 0).long(),                  # [K,H,W]
            "class_id": item["class_id"],
            "subcls": item["subcls"],
        }


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def poly_lr(optimizer, base_lr, current_iter, max_iter, power=0.9, warmup=False, warmup_step=0):
    """Poly learning rate schedule (same as SAM-RSP original)."""
    if warmup and current_iter < warmup_step:
        lr = base_lr * (current_iter + 1) / warmup_step
    else:
        lr = base_lr * (1 - float(current_iter) / float(max_iter)) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_fb_iou(pred: np.ndarray, target: np.ndarray) -> dict:
    """Foreground-Background IoU.

    Args:
        pred: [H, W] binary prediction (1=FG, 0=BG)
        target: [H, W] binary ground truth
    """
    fg_pred = pred == 1
    fg_gt = target == 1
    bg_pred = ~fg_pred
    bg_gt = ~fg_gt

    fg_inter = (fg_pred & fg_gt).sum()
    fg_union = (fg_pred | fg_gt).sum()
    bg_inter = (bg_pred & bg_gt).sum()
    bg_union = (bg_pred | bg_gt).sum()

    fg_iou = fg_inter / max(fg_union, 1)
    bg_iou = bg_inter / max(bg_union, 1)
    return {"FG-IoU": fg_iou, "BG-IoU": bg_iou, "FB-IoU": (fg_iou + bg_iou) / 2}


def train(args: argparse.Namespace) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    fold_def = ISAID5I_FOLDS[args.fold]
    base_classes = sorted(fold_def["train"])
    num_base = len(base_classes)
    print(f"[Stage2] fold={args.fold} shot={args.shot} base_classes={base_classes}")

    data_root = Path(args.data_root)

    # ── Data ──
    train_transform = Compose([
        RandomScale((0.5, 2.0)),
        RandomCrop(args.crop_size),
        RandomHorizontalFlip(),
        Normalize(),
    ])
    val_transform = Compose([
        Resize(args.val_size),
        Normalize(),
    ])

    # Full dataset (base classes, train split)
    full_ds = FewShotEpisodeDataset(
        data_root, fold=args.fold, shot=args.shot,
        use_base=True, split="train", seed=args.seed,
    )

    # Split into train/val (90/10, same as Stage 1)
    n_total = len(full_ds)
    n_val = min(n_total // 10, 1000)
    n_train = n_total - n_val
    indices = list(range(n_total))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]

    train_ds = EpisodeTransformWrapper(
        Subset(full_ds, train_indices), train_transform,
    )
    val_ds = EpisodeTransformWrapper(
        Subset(full_ds, val_indices), val_transform,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    print(f"[Stage2] train={n_train} val={n_val} (from {n_total} total)")

    # ── Model ──
    model = BAMModel(
        num_base_classes=num_base,
        shot=args.shot,
        layers=50,
        zoom_factor=8,
        low_fea='layer2',
        kshot_trans_dim=2,
    ).to(device)

    if args.stage1_ckpt:
        model.load_stage1_weights(args.stage1_ckpt)
    model.freeze_backbone_and_base()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[Stage2] params={n_params:.1f}M trainable={n_trainable:.1f}M")

    # ── Optimizer ──
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = SGD(
        model.get_trainable_params(),
        lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
    )

    # ── Output ──
    out_dir = Path(args.output_dir) / f"fold{args.fold}_shot{args.shot}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_model.pth"
    best_fb_iou = -1.0

    max_iter = args.epochs * len(train_loader)

    for epoch in range(args.epochs):
        # ── Train ──
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for i, batch in enumerate(pbar):
            current_iter = epoch * len(train_loader) + i + 1
            poly_lr(optimizer, args.lr, current_iter, max_iter, power=0.9)

            q_img = batch["query_image"].to(device)
            q_mask = batch["query_mask"].to(device)
            s_img = batch["support_images"].to(device)
            s_mask = batch["support_masks"].to(device)
            subcls = batch["subcls"].to(device)  # [B]

            final_out, base_out = model(q_img, s_img, s_mask, subcls)

            # 2-class CE: FG/BG
            loss = criterion(
                final_out,
                q_mask.clamp(0, 1),  # binary mask (1=FG, 0=BG)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        train_loss /= max(len(train_loader), 1)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        fg_inter = 0
        fg_union = 0
        bg_inter = 0
        bg_union = 0
        with torch.no_grad():
            for batch in val_loader:
                q_img = batch["query_image"].to(device)
                q_mask = batch["query_mask"].to(device)
                s_img = batch["support_images"].to(device)
                s_mask = batch["support_masks"].to(device)
                subcls = batch["subcls"].to(device)

                final_out, base_out = model(q_img, s_img, s_mask, subcls)

                loss = criterion(
                    final_out,
                    q_mask.clamp(0, 1),
                )
                val_loss += loss.item()

                # FG/BG IoU
                pred = final_out.argmax(1).cpu().numpy()  # [B, H, W]
                target = q_mask.clamp(0, 1).cpu().numpy()

                for b in range(pred.shape[0]):
                    pred_b = pred[b]
                    target_b = target[b]
                    fg_inter += ((pred_b == 1) & (target_b == 1)).sum()
                    fg_union += ((pred_b == 1) | (target_b == 1)).sum()
                    bg_inter += ((pred_b == 0) & (target_b == 0)).sum()
                    bg_union += ((pred_b == 0) | (target_b == 0)).sum()

        val_loss /= max(len(val_loader), 1)
        fg_iou = fg_inter / max(fg_union, 1)
        bg_iou = bg_inter / max(bg_union, 1)
        fb_iou = (fg_iou + bg_iou) / 2

        print(f"[{epoch:>3d}] train_loss={train_loss:.3f} val_loss={val_loss:.3f} "
              f"FG-IoU={fg_iou:.4f} BG-IoU={bg_iou:.4f} FB-IoU={fb_iou:.4f}")

        if fb_iou > best_fb_iou:
            best_fb_iou = fb_iou
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "fb_iou": fb_iou,
                "fg_iou": fg_iou,
                "bg_iou": bg_iou,
                "num_base_classes": num_base,
                "base_classes": base_classes,
                "shot": args.shot,
                "fold": args.fold,
            }, best_path)
            print(f"  -> best: FB-IoU={fb_iou:.4f}")

    print(f"\n[Stage2] Done. best_FB-IoU={best_fb_iou:.4f} saved to {best_path}")
    return best_path


def main() -> None:
    p = argparse.ArgumentParser(description="SAM-RSP Stage 2: BAM Training")
    p.add_argument("--fold", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--shot", type=int, default=1, help="K-shot setting")
    p.add_argument("--data-root", type=str, default=str(_REPO_ROOT / "data" / "iSAID-5i"))
    p.add_argument("--stage1-ckpt", type=str, required=True,
                   help="Path to Stage 1 best_model.pth")
    p.add_argument("--output-dir", type=str, default=str(_REPO_ROOT / "runs" / "sam_rsp_stage2"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--crop-size", type=int, default=473)
    p.add_argument("--val-size", type=int, default=473)
    p.add_argument("--seed", type=int, default=321)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    if not Path(args.stage1_ckpt).exists():
        print(f"ERROR: Stage 1 checkpoint not found: {args.stage1_ckpt}")
        print("Run Stage 1 first: python tools/sam_rsp_stage1.py --fold 0 --epochs 100")
        sys.exit(1)

    train(args)


if __name__ == "__main__":
    main()
