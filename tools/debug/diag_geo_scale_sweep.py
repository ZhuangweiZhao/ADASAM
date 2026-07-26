"""
GeoPrior Scale Sweep — 因果验证: Geo 的能量是否压垮了 PromptFusion？
=====================================================================

两个关键实验:

  实验1: Scale Sweep
    将 Geo 缩放 α 倍再与 SPG 拼接送入 PF:
      dense = PF(concat(α·Geo, SPG))
    α ∈ {0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10}
    测量: mIoU, PF 输出 K90, Channel Gini

    如果存在明显最优 α ≠ 1 → 能量失衡直接证实
    如果 α=0 (只用SPG) 最好 → Geo 就是噪声
    如果 α=1 最好 → 能量不是问题, PF 结构是问题

  实验2: Normalization Ablation
    Geo_norm = LayerNorm(Geo) 或 L2Norm(Geo)
    dense = PF(concat(Geo_norm, SPG))
    测量: mIoU, PF 输出 K90

用法:
    python tools/debug/diag_geo_scale_sweep.py \
        --stage2-ckpt runs/stage2_fold1_k5_seed42/best_model.pt \
        --data-root data/iSAID-5i --fold 1 --mode novel --k-shot 5 \
        --num-tiles 50
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    eval_iou,
    print_header,
    save_results,
)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


@torch.no_grad()
def compute_effective_rank(prompt: torch.Tensor) -> float:
    C = prompt.shape[1]
    flat = prompt[0].reshape(C, -1).float()
    _, S, _ = torch.linalg.svd(flat, full_matrices=False)
    s2 = S ** 2
    cumsum = torch.cumsum(s2, dim=0) / (s2.sum() + 1e-10)
    return float((cumsum < 0.90).sum() + 1)


@torch.no_grad()
def run_pf_with_geo(ctx: DiagContext, geo_raw, spg_raw, geo_transform: str = "none",
                     alpha: float = 1.0) -> torch.Tensor:
    """Run PromptFusion with a modified GeoPrior.

    :param geo_raw: raw geometric_prior [1, 256, 64, 64]
    :param spg_raw: raw semantic_prior [1, 256, 64, 64]
    :param geo_transform: "none" | "scale" | "layernorm" | "l2norm"
    :param alpha: scale factor (used when transform="scale")
    :return: dense_prompt [1, 256, 64, 64]
    """
    model = ctx.model

    # Transform Geo
    if geo_transform == "none":
        geo = geo_raw
    elif geo_transform == "scale":
        geo = geo_raw * alpha
    elif geo_transform == "layernorm":
        # LayerNorm over channel+spatial (per-sample)
        geo_flat = geo_raw.reshape(1, -1)
        mean = geo_flat.mean(dim=1, keepdim=True)
        std = geo_flat.std(dim=1, keepdim=True) + 1e-5
        geo = ((geo_raw.reshape(1, -1) - mean) / std).reshape_as(geo_raw)
    elif geo_transform == "l2norm":
        # L2 normalize to unit norm (per-sample, all elements)
        norm = geo_raw.norm(p=2) + 1e-10
        geo = geo_raw / norm
    elif geo_transform == "channel_layernorm":
        # LayerNorm per-channel (normalize each channel independently)
        C = geo_raw.shape[1]
        geo = geo_raw.clone()
        for c in range(C):
            ch = geo[0, c]
            mean_c = ch.mean()
            std_c = ch.std() + 1e-5
            geo[0, c] = (ch - mean_c) / std_c
    elif geo_transform == "channel_l2norm":
        # L2 normalize each channel independently
        C = geo_raw.shape[1]
        geo = geo_raw.clone()
        for c in range(C):
            ch_norm = geo[0, c].norm(p=2) + 1e-10
            geo[0, c] = geo[0, c] / ch_norm
    else:
        raise ValueError(f"Unknown transform: {geo_transform}")

    # Run PromptFusion
    dense_prompt, _ = model.prompt_fusion(geo, spg_raw)
    return dense_prompt


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="GeoPrior Scale Sweep — causal test of energy imbalance"
    )
    add_common_args(parser)
    parser.set_defaults(num_tiles=50)
    parser.add_argument("--alphas", nargs="+", type=float,
                       default=[0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10],
                       help="Scale factors for GeoPrior")
    args = parser.parse_args()

    ctx = build_diag_context(args, require_adapter=True, split="val")
    build_support_cache(ctx)
    select_tiles(ctx, num_tiles=args.num_tiles)

    # ── Experiment 1: Scale Sweep ──
    alpha_results = defaultdict(list)  # {alpha: [{iou, k90, gini}, ...]}

    # ── Experiment 2: Normalization Ablation ──
    norm_results = defaultdict(list)  # {norm_name: [{iou, k90, gini}, ...]}

    norm_methods = ["layernorm", "l2norm", "channel_layernorm", "channel_l2norm"]

    for tile_idx, present_classes in tqdm(ctx.selected_tiles, desc="geo sweep"):
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

        model = ctx.model
        with torch.no_grad():
            support_memory = model.support_encoder(sup_feat, sup_mask)

            geometric_prior = model.geometric_prior(query_emb, support_memory)
            dense_pe = model.sam_decoder.prompt_encoder.get_dense_pe()
            spg_out = model.spg(query_emb, support_memory, dense_pe)
            semantic_prior = spg_out.semantic_prior

        # ── Baseline (current PF) ──
        baseline_prompt, _ = model.prompt_fusion(geometric_prior, semantic_prior)
        baseline_iou = eval_iou(model, baseline_prompt, query_emb, sup_feat, sup_mask, gt, H, W)
        baseline_k90 = compute_effective_rank(baseline_prompt)

        # ── Scale Sweep ──
        for alpha in args.alphas:
            dense_prompt = run_pf_with_geo(ctx, geometric_prior, semantic_prior,
                                           geo_transform="scale", alpha=alpha)
            iou_val = eval_iou(model, dense_prompt, query_emb, sup_feat, sup_mask, gt, H, W)
            k90_val = compute_effective_rank(dense_prompt)
            ch_energy = dense_prompt[0].pow(2).reshape(256, -1).mean(dim=1)
            sorted_e = torch.sort(ch_energy)[0]
            n = len(sorted_e)
            idx = torch.arange(1, n + 1, device=sorted_e.device, dtype=torch.float32)
            gini = float(1 - 2 * torch.sum(sorted_e * idx) / (n * sorted_e.sum() + 1e-10) + (n + 1) / n)

            alpha_results[alpha].append({
                "iou": iou_val, "k90": k90_val, "gini": gini,
            })

        # Also include SPG-only (no Geo at all) as alpha=-1 marker
        spg_only_iou = eval_iou(model, semantic_prior, query_emb, sup_feat, sup_mask, gt, H, W)
        spg_only_k90 = compute_effective_rank(semantic_prior)
        alpha_results[-1].append({
            "iou": spg_only_iou, "k90": spg_only_k90, "gini": 0.0,
            "label": "SPG-only (skip PF entirely)"
        })

        # ── Normalization Ablation ──
        for norm_name in norm_methods:
            try:
                dense_prompt = run_pf_with_geo(ctx, geometric_prior, semantic_prior,
                                               geo_transform=norm_name)
                iou_val = eval_iou(model, dense_prompt, query_emb, sup_feat, sup_mask, gt, H, W)
                k90_val = compute_effective_rank(dense_prompt)
                norm_results[norm_name].append({"iou": iou_val, "k90": k90_val})
            except Exception as e:
                print(f"  WARN: {norm_name} failed on tile {tile_idx}: {e}")

        # Baseline stats
        norm_results["baseline"].append({
            "iou": baseline_iou, "k90": baseline_k90,
        })

    # ── Aggregate ──
    print_header("GeoPrior Scale Sweep — Causal Energy Imbalance Test")

    # Scale sweep summary
    print("\n  ┌─ Scale Sweep: IoU vs α·Geo{'─'*47}")
    print(f"  {'α':>8s}  {'mIoU':>10s}  {'K90':>8s}  {'Gini':>8s}  {'ΔIoU':>10s}")
    print("  " + "-" * 56)

    baseline_mean_iou = np.mean([d["iou"] for d in norm_results["baseline"]])

    alpha_summary = {}
    best_alpha = None
    best_iou = -1

    for alpha in sorted(alpha_results.keys()):
        items = alpha_results[alpha]
        ious = [d["iou"] for d in items]
        k90s = [d["k90"] for d in items]
        ginis = [d.get("gini", 0) for d in items]
        mean_iou = np.mean(ious)
        mean_k90 = np.mean(k90s)
        mean_gini = np.mean(ginis)
        delta = mean_iou - baseline_mean_iou

        label = f"α={alpha:.3g}" if alpha >= 0 else "SPG-only"
        print(f"  {label:>8s}  {mean_iou:>10.4f}  {mean_k90:>8.1f}  {mean_gini:>8.3f}  {delta:>+10.4f}")

        alpha_summary[alpha] = {
            "mean_iou": round(mean_iou, 4),
            "std_iou": round(float(np.std(ious)), 4),
            "mean_k90": round(mean_k90, 2),
            "mean_gini": round(mean_gini, 4),
            "delta_iou": round(delta, 4),
            "n_tiles": len(ious),
        }

        if alpha >= 0 and mean_iou > best_iou:
            best_iou = mean_iou
            best_alpha = alpha

    # Best alpha verdict
    print(f"\n  ★ Best α = {best_alpha}, IoU = {best_iou:.4f} (baseline={baseline_mean_iou:.4f})")
    if best_alpha < 0.5:
        print(f"    → Geo's energy IS overwhelming SPG. Suppressing Geo helps.")
    elif best_alpha > 2:
        print(f"    → Geo is underutilized. Boosting Geo helps.")
    else:
        print(f"    → α near 1 is optimal. Energy imbalance is NOT the primary issue.")

    # ── Normalization Ablation ──
    print("\n\n  ┌─ Normalization Ablation{'─'*54}")
    print(f"  {'Method':>25s}  {'mIoU':>10s}  {'K90':>8s}  {'ΔIoU':>10s}")
    print("  " + "-" * 65)

    for norm_name in ["baseline"] + norm_methods:
        items = norm_results[norm_name]
        ious = [d["iou"] for d in items]
        k90s = [d["k90"] for d in items]
        mean_iou = np.mean(ious)
        mean_k90 = np.mean(k90s)
        delta = mean_iou - baseline_mean_iou
        print(f"  {norm_name:>25s}  {mean_iou:>10.4f}  {mean_k90:>8.1f}  {delta:>+10.4f}")

    # ── Save ──
    summary = {
        "baseline_iou": round(baseline_mean_iou, 4),
        "best_alpha": best_alpha,
        "best_alpha_iou": round(best_iou, 4),
        "scale_sweep": {str(k): v for k, v in alpha_summary.items()},
        "norm_ablation": {
            name: {
                "mean_iou": round(float(np.mean([d["iou"] for d in items])), 4),
                "mean_k90": round(float(np.mean([d["k90"] for d in items])), 2),
                "n": len(items),
            }
            for name, items in norm_results.items()
        },
    }

    s_path, p_path = save_results(ctx.out_dir, summary, [])
    print(f"\n[Summary] {s_path}")
    print("[Done]")


if __name__ == "__main__":
    main()
