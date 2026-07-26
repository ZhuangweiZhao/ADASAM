"""
Quick comparison: PromptFusion vs SPG-only vs Geo-only prompt IoU
===================================================================

Tests whether bypassing PromptFusion (using SPG semantic_prior directly)
improves decoder IoU, based on the PCA finding that PromptFusion collapses
256 channels → 2 effective dimensions.

Usage:
    python tools/debug/compare_prompt_variants.py \
        --stage2-ckpt runs/stage2_fold1_k5_seed42/best_model.pt \
        --data-root data/iSAID-5i --fold 1 --mode novel --k-shot 5 \
        --num-tiles 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adasam.datasets.isaid_5i import ISAID5I_CATEGORIES
from adasam.utils.transforms import preprocess_image

from tools.analysis._decoder_diag_base import (
    DiagContext,
    add_common_args,
    build_diag_context,
    build_support_cache,
    select_tiles,
    extract_dense_prompt,
    eval_iou,
    print_header,
    print_metric_table,
    aggregate_metrics,
    save_results,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compare PromptFusion vs SPG-only vs Geo-only prompt IoU"
    )
    add_common_args(parser)
    parser.add_argument("--num-tiles", type=int, default=50)
    args = parser.parse_args()

    ctx = build_diag_context(args, require_adapter=True, split="val")
    build_support_cache(ctx)
    select_tiles(ctx, num_tiles=args.num_tiles)

    per_tile = []

    for tile_idx, present_classes in tqdm(ctx.selected_tiles, desc="compare"):
        sample = ctx.dataset[tile_idx]
        H, W = sample["image"].shape[1], sample["image"].shape[2]

        xx, _ = preprocess_image(sample["image"])
        query_emb = ctx.backbone(xx.unsqueeze(0).to(ctx.device))["image_embedding"]
        if ctx.adapter is not None:
            query_emb = ctx.adapter(query_emb)

        main_cls = max(present_classes, key=lambda c:
                       ctx.dataset.get_class_mask(tile_idx, c).sum())
        gt = ctx.dataset.get_class_mask(tile_idx, main_cls).numpy().astype(bool)
        sup_data = ctx.support_cache.get(main_cls)
        if sup_data is None:
            continue
        sup_feat, sup_mask = sup_data

        # Extract all prompt variants
        dense_prompt, semantic_prior, geometric_prior = extract_dense_prompt(
            ctx, query_emb, sup_feat, sup_mask
        )

        # Evaluate each variant
        iou_pf = eval_iou(ctx.model, dense_prompt, query_emb, sup_feat, sup_mask, gt, H, W)
        iou_spg = eval_iou(ctx.model, semantic_prior, query_emb, sup_feat, sup_mask, gt, H, W)
        iou_geo = 0.0
        if geometric_prior is not None:
            iou_geo = eval_iou(ctx.model, geometric_prior, query_emb, sup_feat, sup_mask, gt, H, W)

        # Also test: equal-weight fusion (no learned PromptFusion)
        if geometric_prior is not None:
            simple_fusion = (semantic_prior + geometric_prior) / 2.0
            iou_simple_fusion = eval_iou(ctx.model, simple_fusion, query_emb, sup_feat, sup_mask, gt, H, W)
        else:
            iou_simple_fusion = iou_spg

        entry = {
            "tile_idx": tile_idx,
            "class_id": main_cls,
            "class_name": ISAID5I_CATEGORIES.get(main_cls, f"cls{main_cls}"),
            "iou_promptfusion": iou_pf,
            "iou_spg_only": iou_spg,
            "iou_geo_only": iou_geo,
            "iou_simple_fusion": iou_simple_fusion,
            "delta_spg_vs_pf": iou_spg - iou_pf,
        }
        per_tile.append(entry)

    # ── Aggregate ──
    summary = {"n_tiles": len(per_tile)}
    summary["metrics"] = aggregate_metrics(per_tile)

    s_path, p_path = save_results(ctx.out_dir, summary, per_tile)

    # ── Print report ──
    print_header("Prompt Variant Comparison: PromptFusion vs SPG-only")
    if "metrics" in summary:
        print("\n  ┌─ IoU by Prompt Source")
        for key, label in [
            ("iou_promptfusion", "PromptFusion (current)"),
            ("iou_spg_only", "SPG-only (semantic_prior)"),
            ("iou_geo_only", "Geo-only (geometric_prior)"),
            ("iou_simple_fusion", "Simple avg (SPG+Geo)/2"),
        ]:
            if key in summary["metrics"]:
                m = summary["metrics"][key]
                print(f"    {label:<35s} IoU={m['mean']:.4f} ± {m['std']:.4f}")

        if "delta_spg_vs_pf" in summary["metrics"]:
            m = summary["metrics"]["delta_spg_vs_pf"]
            print(f"\n  ★ SPG-only − PromptFusion ΔIoU = {m['mean']:+.4f} ± {m['std']:.4f}")
            n_better = sum(1 for t in per_tile if t.get("delta_spg_vs_pf", 0) > 0.01)
            print(f"    SPG better on {n_better}/{len(per_tile)} tiles ({100*n_better/max(len(per_tile),1):.0f}%)")

    print(f"\n[Summary] {s_path}")
    print(f"[Per-tile] {p_path}")
    print("[Done]")


if __name__ == "__main__":
    main()
