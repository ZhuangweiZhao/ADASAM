"""
Support Sensitivity Test — 同一 Query 不同 Support 的预测对比
============================================================

核心问题: 模型是在做 Class-Conditioned Segmentation 还是 Objectness Detection?

实验设计:
    固定 query tile, 分别用 support=class_A, support=class_B, support=class_C 预测,
    生成 RGB overlay (R=pred_A, G=pred_B, B=pred_C) + difference maps。

判定标准:
    - 情况 A (Objectness): 三张预测几乎相同 → RGB overlay 呈白色/灰色
      → 模型对所有 support class 输出相同的 "前景区域"
    - 情况 B (Class-Conditioned): 三张预测明显不同 → RGB overlay 呈红/绿/蓝色斑
      → 模型知道 support 是哪个类, 输出该类对应的区域

用法 | Usage::

    # 自动选择包含 ≥3 个类的 query tiles
    python tools/debug/visualize_support_switch.py \\
        --checkpoint runs/stage2_fold1_k5_seed42/best_model.pt \\
        --mode novel --k-shot 5

    # 指定 query tile 和类别
    python tools/debug/visualize_support_switch.py \\
        --checkpoint runs/stage2_fold1_k5_seed42/best_model.pt \\
        --mode novel --k-shot 5 \\
        --tile-idx 42 --classes 1 2 4

    # 多张 query tiles
    python tools/debug/visualize_support_switch.py \\
        --checkpoint runs/stage2_fold1_k5_seed42/best_model.pt \\
        --mode novel --k-shot 5 --num-tiles 8 \\

输出:
    - support_switch/{tile_id}.png: 每 tile 一张大图
      [Query | GT_overlay | pred_A | pred_B | pred_C | RGB_Overlay | Diff_AB | Diff_BC]
    - support_switch/summary.json: 定量统计 (IoU per support class, overlap ratios)

Reference: This experiment is designed to definitively answer whether the model
is doing class-conditioned prediction (weak signal → poor mask quality) or pure
objectness detection (same output regardless of support class).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adasam.adapters import CATAdapter
from adasam.backbone import build_mobile_sam, MobileSAMBackbone
from adasam.datasets.isaid_5i import (
    ISAID5iDataset,
    ISAID5I_CATEGORIES,
)
from adasam.model import AdaSAMModel, AdaSAMModelConfig
from adasam.utils import set_seed
from adasam.utils.transforms import preprocess_image, resize_mask


# ═══════════════════════════════════════════════════════════════════
# Support Cache (per-class encoding, same as eval.py)
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def build_support_for_class(
    *,
    data_root: Path,
    fold: int,
    mode: str,
    class_id: int,
    k_shot: int,
    backbone: MobileSAMBackbone,
    cat_adapter: CATAdapter | None,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """为单个类别构建 support (features, masks)."""
    ds = ISAID5iDataset(root=str(data_root), fold=fold, split="train", mode=mode)
    tile_indices = ds.class_to_tiles(class_id)
    if not tile_indices:
        return None

    # Group by source image for scene-disjoint sampling
    scenes: dict[str, list[int]] = defaultdict(list)
    for idx in tile_indices:
        src = ds._source_images.get(ds.tile_ids[idx], ds.tile_ids[idx])
        scenes[src].append(idx)

    rng = random.Random(seed)
    scene_keys = list(scenes)
    k = min(k_shot, len(scene_keys))
    chosen_scenes = rng.sample(scene_keys, k)

    images, masks = [], []
    for sid in chosen_scenes:
        idx = rng.choice(scenes[sid])
        sample = ds[idx]
        fg = ds.get_class_mask(idx, class_id)
        if fg is None or fg.sum() < 1:
            continue
        x, _ = preprocess_image(sample["image"])
        images.append(x.to(device))
        masks.append(fg)

    if not images:
        return None

    feats = backbone(torch.stack(images, dim=0))["image_embedding"]
    if cat_adapter is not None:
        feats = cat_adapter(feats)

    masks_grid = torch.stack(
        [resize_mask(m, (feats.shape[2], feats.shape[3])).to(device) for m in masks],
        dim=0,
    )
    return feats, masks_grid


# ═══════════════════════════════════════════════════════════════════
# Colormap (16 classes, consistent with eval.py vis)
# ═══════════════════════════════════════════════════════════════════

COLORS = np.array([
    [0, 0, 0],        # 0: BG (black)
    [255, 0, 0],      # 1: red
    [0, 255, 0],      # 2: green
    [0, 0, 255],      # 3: blue
    [255, 255, 0],    # 4: yellow
    [255, 0, 255],    # 5: magenta
    [0, 255, 255],    # 6: cyan
    [128, 0, 0],      # 7: dark red
    [0, 128, 0],      # 8: dark green
    [0, 0, 128],      # 9: dark blue
    [128, 128, 0],    # 10: olive
    [128, 0, 128],    # 11: purple
    [0, 128, 128],    # 12: teal
    [192, 192, 192],  # 13: silver
    [64, 64, 64],     # 14: dark gray
    [255, 128, 0],    # 15: orange
], dtype=np.uint8)

# RGB channel colors for the 3 support classes in overlay
CHANNEL_COLORS = [
    np.array([255, 0, 0], dtype=np.uint8),     # R: class A
    np.array([0, 255, 0], dtype=np.uint8),     # G: class B
    np.array([0, 128, 255], dtype=np.uint8),   # B: class C
]


def _colorize(label: np.ndarray) -> np.ndarray:
    """Convert class label map to color image."""
    H, W = label.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for c in range(len(COLORS)):
        out[label == c] = COLORS[c]
    return out


def _make_rgb_overlay(masks: list[np.ndarray]) -> np.ndarray:
    """Combine 3 binary masks into an RGB overlay.

    R = masks[0], G = masks[1], B = masks[2].
    Overlap regions show mixed colors:
      R+G = yellow, R+B = magenta, G+B = cyan, R+G+B = white.
    """
    H, W = masks[0].shape
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    for i, mask in enumerate(masks):
        if mask is not None and mask.sum() > 0:
            color = CHANNEL_COLORS[i]
            overlay[mask.astype(bool)] = np.clip(
                overlay[mask.astype(bool)].astype(int) + color, 0, 255
            ).astype(np.uint8)
    return overlay


def _compute_overlap_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two binary masks."""
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter / union) if union > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_support_sensitivity_test(
    *,
    checkpoint_path: Path,
    data_root: Path,
    fold: int,
    mode: str,
    k_shot: int,
    seed: int,
    score_thr: float,
    device: torch.device,
    tile_indices: list[int] | None = None,
    class_ids: list[int] | None = None,
    num_tiles: int = 6,
    out_dir: Path | None = None,
) -> dict:
    """运行 Support Sensitivity Test | Run support sensitivity test.

    :param tile_indices: 指定 query tile 索引列表 (None = 自动选择).
    :param class_ids: 指定测试类别 (None = 自动选择每 tile 上存在的 3 个类).
    :param num_tiles: 当 tile_indices=None 时, 自动选择的 tile 数量.
    :param out_dir: 输出目录 (默认: checkpoint 所在目录/support_switch).
    :return: summary dict with per-tile statistics.
    """
    set_seed(seed)

    # ── Load model ──
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    bb_cfg = cfg.get("backbone", {})
    bb_path = Path(bb_cfg.get("checkpoint", "weights/mobile_sam.pt"))
    bb_path = bb_path if bb_path.is_absolute() else _REPO_ROOT / bb_path

    sam = build_mobile_sam(str(bb_path), bb_cfg.get("model_type", "vit_t"), device)
    backbone = MobileSAMBackbone(sam.image_encoder, sam.image_encoder.img_size).to(device)
    embed_dim = int(cfg.get("support_encoder", {}).get("embed_dim", 256))
    model = AdaSAMModel(sam, AdaSAMModelConfig.from_dict(cfg)).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[WARN] missing keys: {missing}")
    if unexpected:
        print(f"[WARN] unexpected keys: {unexpected}")
    model.eval()

    # CAT-Adapter
    cat_adapter = None
    if ckpt.get("cat_adapter") is not None:
        tcfg = cfg.get("train", {})
        adapter_cfg = tcfg.get("cat_adapter", {})
        cat_adapter = CATAdapter(
            dim=embed_dim, bottleneck=int(adapter_cfg.get("bottleneck", 64)),
        ).to(device)
        cat_adapter.load_state_dict(ckpt["cat_adapter"])
        cat_adapter.eval()

    # ── Data ──
    val_ds = ISAID5iDataset(root=str(data_root), fold=fold, split="val", mode=mode)
    visible_classes = val_ds.visible_classes()

    # ── Select query tiles ──
    if tile_indices is None:
        # Auto-select tiles with the most classes present
        tile_class_counts = []
        for idx in range(len(val_ds)):
            present = []
            for cls in visible_classes:
                gt_m = val_ds.get_class_mask(idx, cls)
                if gt_m is not None and gt_m.sum() > 10:  # at least 10 pixels
                    present.append(cls)
            if len(present) >= 3:
                tile_class_counts.append((idx, len(present), present))

        # Sort by number of classes (descending), pick top N
        tile_class_counts.sort(key=lambda x: -x[1])
        selected = tile_class_counts[:num_tiles]
        print(f"[select] found {len(tile_class_counts)} tiles with ≥3 classes")
        for idx, n_cls, present in selected:
            names = [ISAID5I_CATEGORIES.get(c, f"cls{c}") for c in present[:5]]
            print(f"  tile {idx}: {n_cls} classes — {names}")
    else:
        selected = []
        for idx in tile_indices:
            present = []
            for cls in visible_classes:
                gt_m = val_ds.get_class_mask(idx, cls)
                if gt_m is not None and gt_m.sum() > 10:
                    present.append(cls)
            selected.append((idx, len(present), present))

    if not selected:
        print("[ERROR] no suitable tiles found!")
        return {}

    # ── Output dir ──
    if out_dir is None:
        out_dir = checkpoint_path.parent / "support_switch"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] {out_dir}")

    # ── Per-tile experiment ──
    all_summaries = []

    for tile_idx, n_cls, present_classes in tqdm(selected, desc="support_switch"):
        sample = val_ds[tile_idx]
        query_image = sample["image"]
        H, W = query_image.shape[1], query_image.shape[2]
        img_np = (query_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Embed query once
        x, meta = preprocess_image(query_image)
        query_emb = backbone(x.unsqueeze(0).to(device))["image_embedding"]
        if cat_adapter is not None:
            query_emb = cat_adapter(query_emb)

        # GT overlay (all classes)
        gt_combined = np.zeros((H, W), dtype=np.uint8)
        for cls in present_classes:
            gt_m = val_ds.get_class_mask(tile_idx, cls)
            if gt_m is not None and gt_m.sum() > 0:
                gt_combined[gt_m.numpy().astype(bool)] = cls
        gt_col = _colorize(gt_combined)

        # Pick 3 test classes for RGB overlay
        if class_ids is not None:
            test_classes = [c for c in class_ids if c in visible_classes]
        else:
            test_classes = present_classes[:3]

        if len(test_classes) < 2:
            print(f"  tile {tile_idx}: only {len(test_classes)} valid classes, skipping")
            continue

        # Pad to exactly 3 for consistent visualization
        while len(test_classes) < 3:
            test_classes.append(test_classes[0])  # duplicate last class if needed

        # ── Predict with each support class ──
        predictions: list[np.ndarray | None] = []
        per_class_stats: list[dict] = []

        for i, cls in enumerate(test_classes):
            sup_data = build_support_for_class(
                data_root=data_root, fold=fold, mode=mode,
                class_id=cls, k_shot=k_shot,
                backbone=backbone, cat_adapter=cat_adapter,
                seed=seed + i * 1000, device=device,
            )

            if sup_data is None:
                print(f"  tile {tile_idx} cls {cls}: no support, skipping")
                predictions.append(None)
                per_class_stats.append({"class": cls, "error": "no_support"})
                continue

            sup_feat, sup_mask = sup_data
            try:
                masks_pred, scores = model.predict(
                    query_emb, sup_feat, sup_mask,
                    meta.input_size, (H, W), score_thr=score_thr,
                )
                if masks_pred.shape[0] > 0:
                    pred = masks_pred.cpu().numpy().squeeze(0).astype(bool)
                else:
                    pred = np.zeros((H, W), dtype=bool)
            except Exception as exc:
                print(f"  tile {tile_idx} cls {cls}: predict error — {exc}")
                pred = np.zeros((H, W), dtype=bool)

            predictions.append(pred)

            # Per-class IoU against own GT
            gt_binary = val_ds.get_class_mask(tile_idx, cls)
            if gt_binary is not None:
                gt_np = gt_binary.numpy().astype(bool)
                iou_own = _compute_overlap_ratio(pred, gt_np)
            else:
                iou_own = float("nan")

            per_class_stats.append({
                "class_id": cls,
                "class_name": ISAID5I_CATEGORIES.get(cls, f"cls{cls}"),
                "IoU_vs_own_GT": round(iou_own, 6),
                "pred_area": int(pred.sum()),
            })

        # ── Cross-class overlap analysis ──
        cross_stats = {}
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                if predictions[i] is not None and predictions[j] is not None:
                    iou_ij = _compute_overlap_ratio(predictions[i], predictions[j])
                    key = f"IoU_pred{test_classes[i]}_vs_pred{test_classes[j]}"
                    cross_stats[key] = round(iou_ij, 6)

        # Per-prediction IoU against ALL class GTs
        for i, cls_support in enumerate(test_classes):
            if predictions[i] is None:
                continue
            for cls_gt in present_classes:
                gt_m = val_ds.get_class_mask(tile_idx, cls_gt)
                if gt_m is not None:
                    gt_np = gt_m.numpy().astype(bool)
                    iou_sg = _compute_overlap_ratio(predictions[i], gt_np)
                    key = f"IoU_sup{cls_support}_gt{cls_gt}"
                    cross_stats[key] = round(iou_sg, 6)

        # ── Visualization ──
        n_cols = 8
        fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 4, 4.5))
        axes = axes.flatten()

        # Column titles
        titles = [
            "Query Image",
            "GT (all classes)",
            f"Pred (sup={ISAID5I_CATEGORIES.get(test_classes[0], '?')})",
            f"Pred (sup={ISAID5I_CATEGORIES.get(test_classes[1], '?')})",
        ]
        if len(test_classes) >= 3:
            titles.append(f"Pred (sup={ISAID5I_CATEGORIES.get(test_classes[2], '?')})")
        titles.append("RGB Overlay\n(R=A, G=B, B=C)")
        titles.append("Diff A-B\n(red=A only, blue=B only)")
        titles.append("Diff A-C\n(red=A only, blue=C only)")
        titles = titles[:n_cols]  # ensure exactly n_cols

        # 1. Query Image
        axes[0].imshow(img_np)
        axes[0].set_title(titles[0], fontsize=9)
        axes[0].axis("off")

        # 2. GT overlay
        axes[1].imshow(gt_col)
        axes[1].set_title(titles[1], fontsize=9)
        axes[1].axis("off")

        # 3-5. Individual predictions
        pred_colors = [(255, 100, 100), (100, 255, 100), (100, 150, 255)]
        for i in range(min(3, len(predictions))):
            ax = axes[2 + i]
            ax.imshow(img_np)
            if predictions[i] is not None and predictions[i].sum() > 0:
                overlay = np.zeros((H, W, 4), dtype=np.uint8)
                overlay[predictions[i], :3] = pred_colors[i]
                overlay[predictions[i], 3] = 180
                ax.imshow(overlay)
            ax.set_title(titles[2 + i], fontsize=9)
            ax.axis("off")

        # 6. RGB Overlay (key visualization)
        ax_rgb = axes[5]
        rgb_overlay = _make_rgb_overlay([
            predictions[0] if len(predictions) > 0 else None,
            predictions[1] if len(predictions) > 1 else None,
            predictions[2] if len(predictions) > 2 else None,
        ])
        # Blend with image
        blended = (img_np.astype(float) * 0.3 + rgb_overlay.astype(float) * 0.7).clip(0, 255).astype(np.uint8)
        ax_rgb.imshow(blended)
        # Legend
        legend_text = "R=" + ISAID5I_CATEGORIES.get(test_classes[0], "?")
        legend_text += "\nG=" + ISAID5I_CATEGORIES.get(test_classes[1], "?")
        if len(test_classes) >= 3:
            legend_text += "\nB=" + ISAID5I_CATEGORIES.get(test_classes[2], "?")
        ax_rgb.text(0.02, 0.98, legend_text, transform=ax_rgb.transAxes,
                    fontsize=7, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax_rgb.set_title(titles[5], fontsize=9)
        ax_rgb.axis("off")

        # 7. Diff A-B
        if len(predictions) >= 2 and predictions[0] is not None and predictions[1] is not None:
            diff_ab = np.zeros((H, W, 3), dtype=np.uint8)
            only_a = predictions[0] & ~predictions[1]
            only_b = predictions[1] & ~predictions[0]
            both = predictions[0] & predictions[1]
            diff_ab[only_a] = (255, 0, 0)    # red: only A
            diff_ab[only_b] = (0, 0, 255)    # blue: only B
            diff_ab[both] = (128, 128, 128)  # gray: overlap
            axes[6].imshow(diff_ab)
        axes[6].set_title(titles[6], fontsize=9)
        axes[6].axis("off")

        # 8. Diff A-C
        if len(predictions) >= 3 and predictions[0] is not None and predictions[2] is not None:
            diff_ac = np.zeros((H, W, 3), dtype=np.uint8)
            only_a = predictions[0] & ~predictions[2]
            only_c = predictions[2] & ~predictions[0]
            both = predictions[0] & predictions[2]
            diff_ac[only_a] = (255, 0, 0)    # red: only A
            diff_ac[only_c] = (0, 0, 255)    # blue: only C
            diff_ac[both] = (128, 128, 128)  # gray: overlap
            axes[7].imshow(diff_ac)
        axes[7].set_title(titles[7], fontsize=9)
        axes[7].axis("off")

        # Save
        tile_id = sample.get("tile_id", str(tile_idx))
        fig.suptitle(f"Support Sensitivity: {tile_id}  (fold={fold}, mode={mode}, k={k_shot})",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        out_path = out_dir / f"{tile_id}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {out_path}")

        # ── Summary ──
        tile_summary = {
            "tile_id": tile_id,
            "tile_index": tile_idx,
            "test_classes": [
                {"id": c, "name": ISAID5I_CATEGORIES.get(c, f"cls{c}")}
                for c in test_classes
            ],
            "per_class": per_class_stats,
            "cross_class": cross_stats,
        }
        all_summaries.append(tile_summary)

    # ── Aggregate summary ──
    summary = {
        "checkpoint": str(checkpoint_path),
        "fold": fold, "mode": mode, "k_shot": k_shot,
        "n_tiles": len(all_summaries),
        "tiles": all_summaries,
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved: {summary_path}")

    # ── Print key metrics ──
    print(f"\n{'='*70}")
    print(f"  Support Sensitivity Summary")
    print(f"{'='*70}")

    # Average cross-prediction IoU (the KEY metric)
    cross_ious = []
    own_ious = []
    for ts in all_summaries:
        for k, v in ts["cross_class"].items():
            if k.startswith("IoU_pred") and "_vs_pred" in k:
                cross_ious.append(v)
        for pc in ts["per_class"]:
            if "IoU_vs_own_GT" in pc and pc["IoU_vs_own_GT"] == pc["IoU_vs_own_GT"]:
                own_ious.append(pc["IoU_vs_own_GT"])

    if cross_ious:
        avg_cross_iou = np.mean(cross_ious)
        print(f"  Avg pairwise IoU between predictions (different support): {avg_cross_iou:.4f}")
        print(f"    → If close to 1.0: OBJECTNESS (same output regardless of support)")
        print(f"    → If close to 0.0: CLASS-CONDITIONED (support changes prediction)")
        print(f"    → Actual: {avg_cross_iou:.4f}")

    if own_ious:
        avg_own_iou = np.mean(own_ious)
        print(f"\n  Avg IoU vs own-class GT: {avg_own_iou:.4f}")
        print(f"    → Model's per-class segmentation quality")

    print(f"{'='*70}")

    return summary


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Support Sensitivity Test — 同一 Query 不同 Support 的预测对比"
    )
    p.add_argument("--checkpoint", required=True, help="path to checkpoint .pt file")
    p.add_argument("--data-root", default="data/iSAID-5i")
    p.add_argument("--fold", type=int, default=None, help="fold override (default: from checkpoint)")
    p.add_argument("--mode", default=None, choices=["base", "novel", "all"],
                   help="mode override (default: from checkpoint)")
    p.add_argument("--k-shot", type=int, default=None, help="override k-shot")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--score-thr", type=float, default=0.1)

    # Tile selection
    p.add_argument("--tile-idx", type=int, nargs="+", default=None,
                   help="specific tile indices to test")
    p.add_argument("--classes", type=int, nargs="+", default=None,
                   help="specific classes for RGB overlay (up to 3)")
    p.add_argument("--num-tiles", type=int, default=6,
                   help="number of tiles to auto-select (default: 6)")

    # Output
    p.add_argument("--output-dir", default=None, help="custom output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Load checkpoint metadata
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    fold = args.fold if args.fold is not None else ckpt.get("fold", 0)
    mode = args.mode if args.mode is not None else ckpt.get("mode", "novel")
    k_shot = args.k_shot if args.k_shot is not None else ckpt.get("k_shot", 5)
    del ckpt

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = _REPO_ROOT / data_root

    out_dir = Path(args.output_dir) if args.output_dir else None

    print(f"[setup] checkpoint: {ckpt_path}")
    print(f"[setup] fold={fold}  mode={mode}  k_shot={k_shot}  device={device}")

    run_support_sensitivity_test(
        checkpoint_path=ckpt_path,
        data_root=data_root,
        fold=fold,
        mode=mode,
        k_shot=k_shot,
        seed=args.seed,
        score_thr=args.score_thr,
        device=device,
        tile_indices=args.tile_idx,
        class_ids=args.classes,
        num_tiles=args.num_tiles,
        out_dir=out_dir,
    )

    print("[Done]")


if __name__ == "__main__":
    main()
