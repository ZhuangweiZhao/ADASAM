"""
SAM-RSP Stage 2: BAM (RSPG) Meta-Learner Training on iSAID-5i Base Classes.
=============================================================================

在 Stage 1 PSPNet 权重基础上训练 BAM 元学习器 (Rough Segment Prompt Generator)。
Train the BAM meta-learner with frozen PSPNet backbone + base learner on
few-shot FG/BG episodes (base classes only).

用法 | Usage::

    python tools/sam_rsp/stage2_train.py --fold 0 --shot 1 --epochs 50 \\
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adasam.sam_rsp.bam import BAMModel
from adasam.sam_rsp.dataset import FewShotEpisodeDataset, ISAID5I_FOLDS
from adasam.utils import set_seed
from tools.sam_rsp.common import (
    Compose, RandomScale, RandomCrop, RandomHorizontalFlip,
    Resize, Normalize, EpisodeTransformWrapper,
    poly_lr, compute_fb_iou, IMAGENET_MEAN, IMAGENET_STD,
)


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
