"""
SAM-RSP Stage 3: Full Model Training on iSAID-5i Base Classes.
===============================================================

在 Stage 1 PSPNet + Stage 2 BAM 基础上训练完整 SAM-RSP 模型。
Train the full SAM-RSP model with frozen BAM + frozen SAM ViT-H encoder
and trainable ViT decoder blocks on few-shot FG/BG episodes (base classes).

用法 | Usage::

    # 先下载 SAM 权重
    python tools/download_sam_weight.py

    # 然后训练
    python tools/sam_rsp_stage3.py --fold 0 --shot 1 --epochs 50 \
        --stage2-ckpt runs/sam_rsp_stage2/fold0_shot1/best_model.pth \
        --sam-ckpt weights/sam_vit_h_4b8939.pth

输出 | Output::

    runs/sam_rsp_stage3/fold0_shot1/best_model.pth
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
from adasam.sam_rsp.sam_rsp_model import SAMRSPModel
from adasam.sam_rsp.dataset import FewShotEpisodeDataset, ISAID5I_FOLDS, IMAGENET_MEAN, IMAGENET_STD
from adasam.utils import set_seed


# ═══════════════════════════════════════════════════════════════════
# Transforms (same as Stage 2)
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
# Episode Transform Wrapper
# ═══════════════════════════════════════════════════════════════════

class EpisodeTransformWrapper:
    """Apply transform to query and each support image/mask independently."""

    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        q_image = item["query_image"].numpy().transpose(1, 2, 0)
        q_mask = item["query_mask"].numpy()

        q_image, q_mask = self.transform(q_image, q_mask)

        s_images = item["support_images"].numpy()
        s_masks = item["support_masks"].numpy()
        K = s_images.shape[0]
        s_imgs_out, s_msks_out = [], []
        for k in range(K):
            s_img_k = s_images[k].transpose(1, 2, 0)
            s_msk_k = s_masks[k]
            si, sm = self.transform(s_img_k, s_msk_k)
            s_imgs_out.append(torch.from_numpy(si).permute(2, 0, 1))
            s_msks_out.append(torch.from_numpy(sm))

        return {
            "query_image": torch.from_numpy(q_image).permute(2, 0, 1).float(),
            "query_mask": torch.from_numpy(q_mask).long(),
            "support_images": torch.stack(s_imgs_out, 0).float(),
            "support_masks": torch.stack(s_msks_out, 0).long(),
            "class_id": item["class_id"],
            "subcls": item["subcls"],
        }


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def poly_lr(optimizer, base_lr, current_iter, max_iter, power=0.9):
    lr = base_lr * (1 - float(current_iter) / float(max_iter)) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args: argparse.Namespace) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    fold_def = ISAID5I_FOLDS[args.fold]
    base_classes = sorted(fold_def["train"])
    num_base = len(base_classes)
    print(f"[Stage3] fold={args.fold} shot={args.shot} base_classes={base_classes}")

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

    full_ds = FewShotEpisodeDataset(
        data_root, fold=args.fold, shot=args.shot,
        use_base=True, split="train", seed=args.seed,
    )

    n_total = len(full_ds)
    n_val = min(n_total // 10, 500)
    n_train = n_total - n_val
    indices = list(range(n_total))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]

    train_ds = EpisodeTransformWrapper(Subset(full_ds, train_indices), train_transform)
    val_ds = EpisodeTransformWrapper(Subset(full_ds, val_indices), val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    print(f"[Stage3] train={n_train} val={n_val}")

    # ── Build BAM, load Stage 2 weights ──
    bam = BAMModel(
        num_base_classes=num_base, shot=args.shot,
        layers=50, zoom_factor=8, low_fea='layer2', kshot_trans_dim=2,
    ).to(device)
    if args.stage2_ckpt:
        bam_ckpt = torch.load(args.stage2_ckpt, map_location='cpu', weights_only=False)
        bam_state = bam_ckpt.get('model', bam_ckpt)
        bam.load_state_dict(bam_state, strict=False)
        print(f"[Stage3] BAM weights loaded from {args.stage2_ckpt}")
    bam.eval()

    # ── Build full SAM-RSP model ──
    model = SAMRSPModel(
        bam_model=bam,
        sam_checkpoint=args.sam_ckpt,
        decoder_depth=args.decoder_depth,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[Stage3] params={n_params:.1f}M trainable={n_trainable:.1f}M")

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
        # Keep BAM + SAM encoder in eval mode
        model.bam.eval()
        model.sam_encoder.eval()

        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for i, batch in enumerate(pbar):
            current_iter = epoch * len(train_loader) + i + 1
            poly_lr(optimizer, args.lr, current_iter, max_iter, power=0.9)

            q_img = batch["query_image"].to(device)
            q_mask = batch["query_mask"].to(device)
            s_img = batch["support_images"].to(device)
            s_mask = batch["support_masks"].to(device)
            subcls = batch["subcls"].to(device)

            final_out, aux_outputs, _ = model(q_img, s_img, s_mask, subcls)

            # Main loss
            loss = criterion(final_out, q_mask.clamp(0, 1))

            # Aux loss (deep supervision from inner classifiers)
            aux_weight = 1.0 / max(len(aux_outputs), 1)
            for aux_out in aux_outputs:
                aux_out_up = F.interpolate(
                    aux_out, size=final_out.shape[2:],
                    mode='bilinear', align_corners=True,
                )
                loss = loss + aux_weight * criterion(aux_out_up, q_mask.clamp(0, 1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        train_loss /= max(len(train_loader), 1)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        fg_inter, fg_union = 0, 0
        bg_inter, bg_union = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                q_img = batch["query_image"].to(device)
                q_mask = batch["query_mask"].to(device)
                s_img = batch["support_images"].to(device)
                s_mask = batch["support_masks"].to(device)
                subcls = batch["subcls"].to(device)

                final_out, aux_outputs, _ = model(q_img, s_img, s_mask, subcls)

                loss = criterion(final_out, q_mask.clamp(0, 1))
                for aux_out in aux_outputs:
                    aux_out_up = F.interpolate(
                        aux_out, size=final_out.shape[2:],
                        mode='bilinear', align_corners=True,
                    )
                    loss = loss + aux_weight * criterion(aux_out_up, q_mask.clamp(0, 1))
                val_loss += loss.item()

                pred = final_out.argmax(1).cpu().numpy()
                target = q_mask.clamp(0, 1).cpu().numpy()
                for b in range(pred.shape[0]):
                    pb, tb = pred[b], target[b]
                    fg_inter += ((pb == 1) & (tb == 1)).sum()
                    fg_union += ((pb == 1) | (tb == 1)).sum()
                    bg_inter += ((pb == 0) & (tb == 0)).sum()
                    bg_union += ((pb == 0) | (tb == 0)).sum()

        val_loss /= max(len(val_loader), 1)
        fg_iou = fg_inter / max(fg_union, 1)
        bg_iou = bg_inter / max(bg_union, 1)
        fb_iou = (fg_iou + bg_iou) / 2

        print(f"[{epoch:>3d}] train_loss={train_loss:.3f} val_loss={val_loss:.3f} "
              f"FG-IoU={fg_iou:.4f} BG-IoU={bg_iou:.4f} FB-IoU={fb_iou:.4f}")

        if fb_iou > best_fb_iou:
            best_fb_iou = fb_iou
            # Save decoder only (exclude frozen BAM + SAM to reduce file size)
            decoder_state = {
                k: v for k, v in model.state_dict().items()
                if not k.startswith('bam.') and not k.startswith('sam_encoder.')
            }
            torch.save({
                "epoch": epoch,
                "decoder": decoder_state,
                "optimizer": optimizer.state_dict(),
                "fb_iou": fb_iou,
                "num_base_classes": num_base,
                "base_classes": base_classes,
                "shot": args.shot,
                "fold": args.fold,
            }, best_path)
            print(f"  -> best: FB-IoU={fb_iou:.4f}")

    print(f"\n[Stage3] Done. best_FB-IoU={best_fb_iou:.4f} saved to {best_path}")
    return best_path


def main() -> None:
    p = argparse.ArgumentParser(description="SAM-RSP Stage 3: Full Model Training")
    p.add_argument("--fold", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--shot", type=int, default=1)
    p.add_argument("--data-root", type=str, default=str(_REPO_ROOT / "data" / "iSAID-5i"))
    p.add_argument("--stage2-ckpt", type=str, required=True,
                   help="Path to Stage 2 BAM best_model.pth")
    p.add_argument("--sam-ckpt", type=str, default="",
                   help="Path to SAM ViT-H checkpoint (sam_vit_h_4b8939.pth)")
    p.add_argument("--output-dir", type=str, default=str(_REPO_ROOT / "runs" / "sam_rsp_stage3"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--crop-size", type=int, default=473)
    p.add_argument("--val-size", type=int, default=473)
    p.add_argument("--decoder-depth", type=int, default=3)
    p.add_argument("--seed", type=int, default=321)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    if not Path(args.stage2_ckpt).exists():
        print(f"ERROR: Stage 2 checkpoint not found: {args.stage2_ckpt}")
        sys.exit(1)
    if args.sam_ckpt and not Path(args.sam_ckpt).exists():
        print(f"ERROR: SAM checkpoint not found: {args.sam_ckpt}")
        print("Download it from:")
        print("  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        print(f"  → save to {_REPO_ROOT / 'weights' / 'sam_vit_h_4b8939.pth'}")
        sys.exit(1)

    train(args)


if __name__ == "__main__":
    main()
