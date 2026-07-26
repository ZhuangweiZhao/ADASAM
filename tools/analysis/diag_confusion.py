"""
类别混淆矩阵诊断 | Category Confusion Matrix Diagnostic.
=========================================================

回答核心问题：模型学到的是 "support类别 → query对应类别" 还是 "support → generic foreground"？

实验：
  Exp A: 固定 query，用不同 support 类别 → 预测 mask vs 所有 GT 类别的 IoU
         → 如果 ship support → mask 只对 ship GT 高 IoU，说明模型有类别概念
         → 如果 ship support → mask 对所有 GT 类 IoU 差不多，说明模型只找"前景"

  Exp B: 聚合所有 query tile → 支持类别 × GT类别 混淆矩阵
         → 对角线高 = 模型是 class-specific
         → 对角线≈非对角线 = 模型是 class-agnostic

用法 | Usage:
    python tools/analysis/diag_confusion.py \
        --stage2-ckpt runs/stage2_fold1_k5_seed42/last_model.pt \
        --data-root data/iSAID-5i --fold 1 --k-shot 5
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from adasam.adapters import CATAdapter
from adasam.backbone import build_mobile_sam, MobileSAMBackbone
from adasam.datasets.isaid_5i import ISAID5iDataset, ISAID5I_CATEGORIES
from adasam.model.adasam_model import AdaSAMModel, AdaSAMModelConfig
from adasam.utils import set_seed
from adasam.utils.transforms import preprocess_image, resize_mask


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """IoU between two binary masks."""
    inter = float((pred & gt).sum())
    union = float((pred | gt).sum())
    return inter / max(union, 1.0)


def build_support_for_class(
    dataset, class_id: int, k_shot: int, device: torch.device,
    backbone: MobileSAMBackbone, adapter,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Build support features + masks for a class (FSS protocol, scene-disjoint)."""
    tiles = dataset.class_to_tiles(class_id)
    if len(tiles) < 1:
        return None

    scenes: dict[str, list[int]] = defaultdict(list)
    for idx in tiles:
        tile_id = dataset.tile_ids[idx]
        src = dataset._source_images.get(tile_id, str(idx))
        scenes[src].append(idx)

    chosen = []
    scene_list = list(scenes.keys())
    rng.shuffle(scene_list)
    for src in scene_list:
        if len(chosen) >= k_shot:
            break
        chosen.append(rng.choice(scenes[src]))

    if len(chosen) < 1:
        return None

    images, masks = [], []
    for idx in chosen:
        sample = dataset[idx]
        fg = dataset.get_class_mask(idx, class_id)
        if fg is None:
            continue
        x, _ = preprocess_image(sample["image"])
        images.append(x.to(device))
        masks.append(fg)

    if not images:
        return None

    feats = backbone(torch.stack(images, dim=0))["image_embedding"]
    if adapter is not None:
        feats = adapter(feats)
    masks_grid = torch.stack(
        [resize_mask(m, (feats.shape[2], feats.shape[3])).to(device) for m in masks],
        dim=0,
    )
    if masks_grid.sum() < 1.0:
        return None
    return feats, masks_grid


def compute_pred_mask(
    model: AdaSAMModel,
    query_features: torch.Tensor,
    support_features: torch.Tensor,
    support_masks: torch.Tensor,
) -> np.ndarray:
    """Run full forward pass → predicted binary mask [256, 256]."""
    with torch.no_grad():
        support_memory = model.support_encoder(support_features, support_masks)

        # Geometric Prior
        if model.geometric_prior is not None:
            gp = model.geometric_prior(query_features, support_memory)
        else:
            gp = None

        # SPG
        dense_pe = model.sam_decoder.prompt_encoder.get_dense_pe()
        spg_out = model.spg(query_features, support_memory, dense_pe)

        # PromptFusion
        if model.prompt_fusion is not None and gp is not None:
            dense_prompt, sparse_token = model.prompt_fusion(gp, spg_out.semantic_prior)
        else:
            dense_prompt = model._build_dense_prompt(
                support_memory, support_features, support_masks
            )
            if dense_prompt is None:
                dense_prompt = spg_out.semantic_prior
            sparse_token = dense_prompt.mean(dim=(2, 3))

        # Decode
        if model.bypass_head is not None:
            low_res = model.bypass_head(dense_prompt)
        else:
            low_res, _ = model.sam_decoder(
                query_features, sparse_token, dense_prompt
            )

    return (low_res[0, 0] > 0).cpu().numpy()  # [256, 256] bool


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Category Confusion Matrix Diagnostic"
    )
    parser.add_argument("--stage2-ckpt", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--n-queries", type=int, default=10,
                        help="Number of query tiles with ≥2 classes")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    print(f"Device: {device}")

    # ── Load checkpoint ──
    ckpt_path = Path(args.stage2_ckpt)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    print(f"  epoch={ckpt.get('epoch', '?')}  fold={ckpt.get('fold', '?')}")

    # ── Build model ──
    weights_path = str(_REPO_ROOT / cfg.get("backbone", {}).get(
        "checkpoint", "weights/mobile_sam.pt"))
    sam = build_mobile_sam(
        weights_path, cfg.get("backbone", {}).get("model_type", "vit_t"), device)
    backbone = MobileSAMBackbone(
        sam.image_encoder, sam.image_encoder.img_size).to(device)

    model_cfg = AdaSAMModelConfig.from_dict(cfg)
    model = AdaSAMModel(sam, model_cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # ── Load adapter ──
    adapter_state = ckpt.get("cat_adapter")
    if adapter_state is not None:
        adapter_cfg = ckpt.get("config", {}).get("adapter", {})
        adapter = CATAdapter(
            dim=256,
            bottleneck=int(adapter_cfg.get("bottleneck", 64)),
        ).to(device)
        adapter.load_state_dict(adapter_state)
        adapter.eval()
        for p in adapter.parameters():
            p.requires_grad_(False)
        print("  adapter: loaded + frozen")
    else:
        adapter = None
        print("  adapter: NONE")

    # ── Dataset ──
    data_root = str(_REPO_ROOT / args.data_root) if not Path(args.data_root).is_absolute() \
        else args.data_root
    dataset = ISAID5iDataset(
        root=data_root, fold=args.fold, split="val", mode="base")
    val_classes = sorted(dataset.visible_classes())
    print(f"Val classes ({len(val_classes)}): {val_classes}")
    for cls in val_classes:
        name = ISAID5I_CATEGORIES.get(cls, f"cls{cls}")
        print(f"  {cls:>2d} {name:<20s}: {len(dataset.class_to_tiles(cls))} tiles")

    # ── Find multi-class query tiles ──
    tile_scores = []
    for idx in range(len(dataset)):
        classes_present = []
        for cls in val_classes:
            gt = dataset.get_class_mask(idx, cls)
            if gt is not None and gt.sum() > 50:
                classes_present.append(cls)
        if len(classes_present) >= 2:
            tile_scores.append((idx, len(classes_present), classes_present))

    tile_scores.sort(key=lambda x: -x[1])
    print(f"\nMulti-class tiles available: {len(tile_scores)}")
    n_queries = min(args.n_queries, len(tile_scores))
    selected = tile_scores[:n_queries]

    # ═══════════════════════════════════════════════════════════════
    # EXP A+B: Per-support-class → per-GT-class IoU (Confusion Matrix)
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 80}")
    print("EXP A+B: Support Class → GT Class Response Matrix")
    print("=" * 80)

    # confusion[support_class][gt_class] = list of IoU values
    confusion: dict[int, dict[int, list[float]]] = {
        sc: {gc: [] for gc in val_classes} for sc in val_classes
    }
    # Also track how often each support class produces non-empty predictions
    support_hit_count: dict[int, int] = defaultdict(int)

    for qi, (tile_idx, n_cls_on_tile, cls_on_tile) in enumerate(
            tqdm(selected, desc="queries")):
        sample = dataset[tile_idx]

        # Embed query once
        x, _ = preprocess_image(sample["image"])
        with torch.no_grad():
            q_emb = backbone(x.unsqueeze(0).to(device))["image_embedding"]
            if adapter is not None:
                q_emb = adapter(q_emb)

        # Get all GT masks for this tile
        gt_masks = {}
        for cls in val_classes:
            gt = dataset.get_class_mask(tile_idx, cls)
            if gt is not None:
                gt_masks[cls] = gt.numpy().astype(bool)

        # Try each support class
        for sup_cls in val_classes:
            sup_data = build_support_for_class(
                dataset, sup_cls, args.k_shot, device, backbone, adapter, rng
            )
            if sup_data is None:
                continue

            sup_feat, sup_mask = sup_data
            pred = compute_pred_mask(model, q_emb, sup_feat, sup_mask)

            if pred.sum() > 0:
                support_hit_count[sup_cls] += 1

            # Compute IoU against every GT class
            for gt_cls in val_classes:
                gt = gt_masks.get(gt_cls)
                if gt is None:
                    continue
                iou = mask_iou(pred, gt)
                confusion[sup_cls][gt_cls].append(iou)

    # ═══════════════════════════════════════════════════════════════
    # Print per-query detail for first few queries
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'─' * 80}")
    print("Per-Query Detail (first 3 queries)")
    print("─" * 80)

    for qi, (tile_idx, n_cls_on_tile, cls_on_tile) in enumerate(selected[:3]):
        print(f"\n  Query tile {tile_idx}: GT classes = "
              f"{[ISAID5I_CATEGORIES.get(c, str(c)) for c in cls_on_tile]}")

        sample = dataset[tile_idx]
        x, _ = preprocess_image(sample["image"])
        with torch.no_grad():
            q_emb = backbone(x.unsqueeze(0).to(device))["image_embedding"]
            if adapter is not None:
                q_emb = adapter(q_emb)

        gt_masks = {}
        for cls in val_classes:
            gt = dataset.get_class_mask(tile_idx, cls)
            if gt is not None:
                gt_masks[cls] = gt.numpy().astype(bool)

        for sup_cls in cls_on_tile[:4]:  # test classes present on this tile
            sup_data = build_support_for_class(
                dataset, sup_cls, args.k_shot, device, backbone, adapter, rng
            )
            if sup_data is None:
                continue
            sup_feat, sup_mask = sup_data
            pred = compute_pred_mask(model, q_emb, sup_feat, sup_mask)

            sup_name = ISAID5I_CATEGORIES.get(sup_cls, str(sup_cls))
            # Show IoU against each GT class on this tile
            parts = []
            for gt_cls in cls_on_tile:
                gt = gt_masks.get(gt_cls)
                if gt is None or gt.sum() == 0:
                    continue
                iou = mask_iou(pred, gt)
                gt_name = ISAID5I_CATEGORIES.get(gt_cls, str(gt_cls))
                marker = " ← CORRECT" if gt_cls == sup_cls else ""
                parts.append(f"{gt_name}={iou:.3f}{marker}")
            print(f"    support={sup_name:<20s} → " + "  ".join(parts))

    # ═══════════════════════════════════════════════════════════════
    # Aggregate: Confusion Matrix
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 80}")
    print("CONFUSION MATRIX: Support Class (row) → GT Class (column)")
    print("  (mean IoU, diagonal = correct classification)")
    print("=" * 80)

    # Compute mean IoU for each (support_class, gt_class) pair
    mean_confusion = {}
    for sup_cls in val_classes:
        for gt_cls in val_classes:
            vals = confusion[sup_cls][gt_cls]
            mean_confusion[(sup_cls, gt_cls)] = np.mean(vals) if vals else 0.0

    # Print header
    col_width = 8
    header = f"{'support ↓':>10s}"
    for gc in val_classes:
        name = ISAID5I_CATEGORIES.get(gc, str(gc))[:col_width]
        header += f"  {name:>{col_width}s}"
    print(header)

    # Print rows
    diagonal_vals = []
    offdiag_vals = []
    for sc in val_classes:
        sup_name = ISAID5I_CATEGORIES.get(sc, str(sc))
        row = f"  {sup_name:>8s}"
        for gc in val_classes:
            val = mean_confusion[(sc, gc)]
            row += f"  {val:{col_width}.3f}"
            if sc == gc:
                diagonal_vals.append(val)
            else:
                offdiag_vals.append(val)
        print(row)

    # ── Summary stats ──
    mean_diag = np.mean(diagonal_vals) if diagonal_vals else 0.0
    mean_offdiag = np.mean(offdiag_vals) if offdiag_vals else 0.0
    ratio = mean_diag / max(mean_offdiag, 1e-8)

    print(f"\n{'─' * 60}")
    print(f"  Mean DIAGONAL (correct class):     {mean_diag:.4f}")
    print(f"  Mean OFF-DIAGONAL (wrong class):   {mean_offdiag:.4f}")
    print(f"  Diagonal / Off-diagonal ratio:     {ratio:.2f}×")
    print()

    if ratio < 1.5:
        print(f"  ❌ MODEL IS CLASS-AGNOSTIC")
        print(f"     Support class barely changes which GT class gets high IoU.")
        print(f"     The model finds 'foreground' but doesn't discriminate classes.")
    elif ratio < 3.0:
        print(f"  ⚠️  MODEL IS WEAKLY CLASS-SPECIFIC")
        print(f"     Support class has some effect, but discrimination is weak.")
    else:
        print(f"  ✅ MODEL IS CLASS-SPECIFIC")
        print(f"     Support class strongly determines which GT class gets activated.")

    # ── Per-support-class hit rate ──
    print(f"\n{'─' * 60}")
    print("Per-support-class prediction rate (non-empty mask):")
    for sc in val_classes:
        name = ISAID5I_CATEGORIES.get(sc, str(sc))
        n_attempts = sum(1 for v in confusion[sc][sc] if v >= 0)  # any GT entry
        n_hits = support_hit_count.get(sc, 0)
        rate = n_hits / max(n_attempts, 1)
        print(f"  {name:<20s}: {n_hits}/{n_attempts} non-empty ({rate:.0%})")

    # ── Best GT class per support class ──
    print(f"\n{'─' * 60}")
    print("Best-matching GT class for each support class:")
    for sc in val_classes:
        best_cls = max(val_classes, key=lambda gc: mean_confusion[(sc, gc)])
        best_iou = mean_confusion[(sc, best_cls)]
        diag_iou = mean_confusion[(sc, sc)]
        sup_name = ISAID5I_CATEGORIES.get(sc, str(sc))
        best_name = ISAID5I_CATEGORIES.get(best_cls, str(best_cls))
        marker = "✓ CORRECT" if best_cls == sc else f"✗ WRONG (diag={diag_iou:.3f})"
        print(f"  {sup_name:<20s} → {best_name:<20s}  IoU={best_iou:.3f}  {marker}")


if __name__ == "__main__":
    main()
