"""
Stage 1 数据流诊断 | End-to-end data flow diagnostic.
=====================================================
快速验证: 标注 → GT → 模型输出, 不依赖训练循环。
Quick check: annotation → GT → model output, no training loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from adasam.adapters import CATAdapter
from adasam.backbone import build_mobile_sam, MobileSAMBackbone
from adasam.datasets.isaid_5i import ISAID5iDataset
from adasam.utils.transforms import preprocess_image


def diag(data_root: str, fold: int = 0, weights: str | None = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    print(f"fold = {fold}")

    # ── Dataset ──
    train_ds = ISAID5iDataset(root=data_root, fold=fold, split="train", mode="base")
    val_ds = ISAID5iDataset(root=data_root, fold=fold, split="val", mode="base")

    print(f"\ntrain tiles: {len(train_ds)},  val tiles: {len(val_ds)}")
    print(f"train visible: {sorted(train_ds.visible_classes())}")
    print(f"val   visible: {sorted(val_ds.visible_classes())}")

    class_to_idx = {c: i for i, c in enumerate(sorted(train_ds.visible_classes()))}
    print(f"\nclass_to_idx: {class_to_idx}")
    print(f"num classes (seg_head output channels): {len(class_to_idx)}")

    # ── Check raw annotation values ──
    print("\n── ① Raw annotation check ──")
    ann_dir = Path(data_root) / "iSAID" / "train" / "semantic_png"
    pngs = sorted(ann_dir.glob("*_instance_color_RGB.png"))[:5]
    for png_path in pngs:
        ann = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        if ann is None:
            print(f"  {png_path.name}: FAILED TO READ")
            continue
        unique_vals = sorted(np.unique(ann).tolist())
        print(f"  {png_path.name}: shape={ann.shape} dtype={ann.dtype} unique={unique_vals}")

    # ── Check GT construction for a few tiles ──
    print("\n── ② GT construction check ──")
    for idx in [0, 10, 100]:
        # Use _build_gt approach manually
        gt = torch.full((256, 256), 255, dtype=torch.long)
        present = []
        for cls_id in train_ds.visible_classes():
            mask = train_ds.get_class_mask(idx, cls_id)
            if mask is not None and mask.sum() > 0:
                gt[mask > 0.5] = class_to_idx[cls_id]
                present.append((cls_id, class_to_idx[cls_id], int(mask.sum().item())))

        gt_unique = sorted((gt[gt != 255]).unique().tolist())
        fg_pct = (gt != 255).sum().item() / (256 * 256) * 100
        print(f"  tile[{idx}]: fg_pct={fg_pct:.1f}% gt_classes={gt_unique} "
              f"present={[(c, idx) for c, idx, _ in present]}")

    # ── Check model output + prediction distribution ──
    print("\n── ③ Model forward pass ──")
    weights_path = weights or str(_REPO_ROOT / "weights" / "mobile_sam.pt")
    sam = build_mobile_sam(weights_path, "vit_t", device)
    backbone = MobileSAMBackbone(sam.image_encoder, sam.image_encoder.img_size).to(device)
    adapter = CATAdapter(dim=256, bottleneck=64).to(device)
    seg_head = torch.nn.Conv2d(256, len(class_to_idx), kernel_size=1).to(device)
    torch.nn.init.xavier_uniform_(seg_head.weight)

    backbone.eval()
    adapter.eval()
    seg_head.eval()

    # Run a few samples and check prediction distribution
    pred_counts = torch.zeros(len(class_to_idx))
    for idx in range(min(5, len(val_ds))):
        sample = val_ds[idx]
        x, _ = preprocess_image(sample["image"])
        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            emb = backbone(x)["image_embedding"]
            adapted = adapter(emb)
            logits = torch.nn.functional.interpolate(
                seg_head(adapted), (256, 256), mode="bilinear", align_corners=False
            )
            pred = logits[0].argmax(dim=0).cpu()

        counts = torch.bincount(pred.flatten(), minlength=len(class_to_idx))
        pred_counts += counts.float()
        print(f"  val[{idx}]: pred distribution = {counts.tolist()}  "
              f"(total={counts.sum().item()})")

    total_preds = pred_counts.sum()
    print(f"\n  overall pred distribution (%):")
    for i, cnt in enumerate(pred_counts.tolist()):
        cls_id = next((k for k, v in class_to_idx.items() if v == i), "?")
        print(f"    index {i} (class {cls_id}): {cnt/total_preds*100:.1f}%")

    # ── Check GT class distribution ──
    print("\n── ④ GT class distribution (val, first 50 tiles) ──")
    gt_counts = torch.zeros(len(class_to_idx))
    gt_total = 0
    for idx in range(min(50, len(val_ds))):
        gt = torch.full((256, 256), 255, dtype=torch.long)
        for cls_id in val_ds.visible_classes():
            mask = val_ds.get_class_mask(idx, cls_id)
            if mask is not None and mask.sum() > 0:
                gt[mask > 0.5] = class_to_idx[cls_id]
        fg = gt != 255
        gt_total += fg.sum().item()
        for c in range(len(class_to_idx)):
            gt_counts[c] += (gt == c).sum().item()

    print(f"  total FG pixels in 50 tiles: {gt_total}")
    for i, cnt in enumerate(gt_counts.tolist()):
        cls_id = next((k for k, v in class_to_idx.items() if v == i), "?")
        print(f"    class {cls_id} (index {i}): {cnt/gt_total*100:.1f}% of FG pixels  "
              f"({cnt:.0f} px)")

    print("\n── Done ──")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data" / "iSAID-5i"))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--weights", default=None)
    args = p.parse_args()
    diag(args.data_root, args.fold, args.weights)
