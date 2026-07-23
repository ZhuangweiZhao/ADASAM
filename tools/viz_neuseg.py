"""
NEU_Seg 可视化 (FastSAM-aligned) | Visualization.
==================================================

用法 | Usage::
    python tools/viz_neuseg.py --mode dataset
    python tools/viz_neuseg.py --mode support --k-shot 3
    python tools/viz_neuseg.py --mode predict --checkpoint <ckpt>
"""

from __future__ import annotations

import argparse, random, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch, torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from adasam.backbone import MultiScaleMobileSAMBackbone
from adasam.datasets import NEUSegDataset
from adasam.utils import set_seed
from tools.train_neuseg import PureDecoderP3P4, pad_to_32

CLASS_NAMES = NEUSegDataset.CLASS_NAMES
CLASS_COLORS = {0: [0.5,0.5,0.5], 1: [1.0,0.0,0.0], 2: [0.0,1.0,0.0], 3: [0.0,0.0,1.0]}


def label_to_rgb(label):
    h, w = label.shape; rgb = np.zeros((h,w,3), dtype=np.float32)
    for c, color in CLASS_COLORS.items(): rgb[label==c] = color
    return rgb


def viz_dataset(data_root, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for split in ["train","test"]:
        ds = NEUSegDataset(root=data_root, split=split)
        print(f"{split}: {len(ds)} samples")
        counts = ds.get_class_pixel_counts()
        fig, axes = plt.subplots(1,2,figsize=(14,5))
        names = list(counts.keys()); pixels = list(counts.values())
        axes[0].bar(names, [p/1e6 for p in pixels],
                    color=[CLASS_COLORS[i] for i in range(4)])
        axes[0].set_title(f"Class Distribution — {split}")
        pct = [100*p/sum(pixels) for p in pixels]
        axes[1].pie(pct, labels=names, colors=[CLASS_COLORS[i] for i in range(4)],
                    autopct="%1.1f%%", startangle=90)
        axes[1].set_title(f"Class Share — {split}")
        fig.tight_layout(); fig.savefig(out/f"class_dist_{split}.png", dpi=100)
        plt.close(fig)

        n_show = min(16, len(ds)); rng = random.Random(42)
        indices = rng.sample(range(len(ds)), n_show)
        cols = 4; rows = (n_show+cols-1)//cols
        fig, axes = plt.subplots(rows, cols, figsize=(12,3*rows))
        axes = axes.flatten() if rows > 1 else [axes] if cols==1 else axes
        for i, idx in enumerate(indices):
            s = ds[idx]; img = s["image"].permute(1,2,0).numpy()
            label = s["masks"].squeeze(0).numpy()
            axes[i].imshow(img*0.5+label_to_rgb(label)*0.5)
            axes[i].set_title(s["image_id"], fontsize=7); axes[i].axis("off")
        for i in range(n_show, len(axes)): axes[i].axis("off")
        fig.suptitle(f"NEU_Seg {split} — Sample Grid", fontsize=12)
        fig.tight_layout(); fig.savefig(out/f"samples_{split}.png", dpi=100)
        plt.close(fig)
    print(f"Saved: {out}")


def viz_support_query(data_root, k_shot, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ds = NEUSegDataset(root=data_root, split="train")
    rng = random.Random(42); n_rows = 4
    fig, axes = plt.subplots(n_rows, k_shot+1, figsize=(3*(k_shot+1),3*n_rows))
    for row in range(n_rows):
        indices = rng.sample(range(len(ds)), k_shot+1)
        for j, si in enumerate(indices[:k_shot]):
            s = ds[si]; img = s["image"].permute(1,2,0).numpy()
            axes[row,j].imshow(img*0.5+label_to_rgb(s["masks"].squeeze(0).numpy())*0.5)
            axes[row,j].set_title(f"Support {j+1}\n{s['image_id']}",fontsize=7)
            axes[row,j].axis("off")
        q = ds[indices[k_shot]]
        axes[row,k_shot].imshow(q["image"].permute(1,2,0).numpy()*0.5+
            label_to_rgb(q["masks"].squeeze(0).numpy())*0.5)
        axes[row,k_shot].set_title(f"Query\n{q['image_id']}",fontsize=7)
        axes[row,k_shot].axis("off")
    fig.suptitle(f"Support/Query Pairs (K={k_shot})",fontsize=12)
    fig.tight_layout(); fig.savefig(out/"support_query_pairs.png",dpi=120)
    plt.close(fig); print(f"Saved: {out}")


def viz_predict(checkpoint_path, data_root, device, out_dir, max_samples=8):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    cfg = ckpt.get("config",{}); nc = ckpt.get("num_classes",4)

    mtype = cfg.get("backbone",{}).get("model_type","vit_t")
    bb_p = Path(cfg.get("backbone",{}).get("checkpoint","weights/mobile_sam.pt"))
    bb_p = bb_p if bb_p.is_absolute() else _REPO_ROOT/bb_p
    img_sz = int(cfg.get("backbone",{}).get("img_size", 224))
    backbone = MultiScaleMobileSAMBackbone.build(str(bb_p), mtype, dev, img_size=img_sz)
    backbone.eval()
    d_cfg = cfg.get("decoder",{})
    decoder = PureDecoderP3P4(p3_ch=int(d_cfg.get("p3_channels",128)),
                               p4_ch=int(d_cfg.get("p4_channels",256)),
                               out_ch=nc, mid_ch=int(d_cfg.get("mid_channels",128))).to(dev)
    decoder.load_state_dict(ckpt["decoder_state_dict"],strict=True); decoder.eval()

    ds = NEUSegDataset(root=data_root, split="test")
    rng = random.Random(42)
    indices = rng.sample(range(len(ds)), min(max_samples, len(ds)))

    @torch.no_grad()
    def predict(img):
        img_p, _, _ = pad_to_32(img.unsqueeze(0))
        feats = backbone(img_p.to(dev))
        logits = decoder(feats["stage1"], feats["stage3"])
        H, W = img.shape[1], img.shape[2]
        return F.interpolate(logits, size=(H,W), mode='bilinear', align_corners=False
                            ).argmax(1).squeeze(0).cpu().numpy()

    fig, axes = plt.subplots(len(indices), 4, figsize=(14,3.5*len(indices)))
    if len(indices)==1: axes = axes.reshape(1,-1)
    for row, idx in enumerate(indices):
        s = ds[idx]; img = s["image"]; gt = s["masks"].squeeze(0).numpy()
        pred = predict(img)
        axes[row,0].imshow(img.permute(1,2,0).numpy())
        axes[row,0].set_title("Original",fontsize=9); axes[row,0].axis("off")
        axes[row,1].imshow(label_to_rgb(gt))
        axes[row,1].set_title("Ground Truth",fontsize=9); axes[row,1].axis("off")
        axes[row,2].imshow(label_to_rgb(pred))
        axes[row,2].set_title("Prediction",fontsize=9); axes[row,2].axis("off")
        diff = np.zeros((*gt.shape,3), dtype=np.float32)
        diff[gt==pred] = [0.5,0.5,0.5]; diff[gt!=pred] = [1.0,0.0,0.0]
        axes[row,3].imshow(diff)
        axes[row,3].set_title(f"Diff\n{s['image_id']}",fontsize=9); axes[row,3].axis("off")
    fig.suptitle(f"Predictions\n{Path(checkpoint_path).parent.name}",fontsize=12)
    fig.tight_layout(); fig.savefig(out/"predictions.png",dpi=120)
    plt.close(fig); print(f"Saved: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="dataset", choices=["dataset","support","predict","all"])
    p.add_argument("--data-root", default="data/NEU_Seg")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--k-shot", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="runs/viz_neuseg")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in ("dataset","all"): viz_dataset(args.data_root, str(out_dir/"dataset"))
    if args.mode in ("support","all"): viz_support_query(args.data_root, args.k_shot, str(out_dir/"support"))
    if args.mode in ("predict","all"):
        if not args.checkpoint: print("[ERROR] --checkpoint required"); sys.exit(1)
        viz_predict(args.checkpoint, args.data_root, args.device, str(out_dir/"predict"))
    print(f"\n[Done] {out_dir}")


if __name__ == "__main__": main()
