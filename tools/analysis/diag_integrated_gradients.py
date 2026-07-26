"""
Decoder Integrated Gradients — Prompt Channel Attribution via Gradient
======================================================================

核心问题: Decoder 的预测对 prompt 每个 channel 的梯度贡献有多大？
这是因果归因 (causal attribution), 不是相关性 (Pearson r)。

方法:
  1. 从 zero baseline (α=0) 线性插值到 actual prompt (α=1), m=50 步
  2. 每一步: 计算 ∂(prediction_FG_logit) / ∂(prompt), 对 spatial 聚合
  3. IG(channel_i) = prompt_i × mean_path(∂loss/∂prompt_i)
  4. 同时在 channel 维度聚合得到 spatial IG map (哪个空间位置最重要)

对比:
  - IG score vs GT-Pearson-r: IG 是因果的, Pearson 是相关的
  - IG Top-K vs Pearson Top-K: 哪个更好地保留预测能力？

用法:
    python tools/analysis/diag_integrated_gradients.py \
        --stage2-ckpt runs/stage2_fold1_k5_seed42/best_model.pt \
        --data-root data/iSAID-5i --fold 1 --mode novel --k-shot 5 \
        --num-tiles 20 --ig-steps 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
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
    extract_dense_prompt,
    print_header,
    print_metric_table,
    aggregate_metrics,
    save_results,
)


# ═══════════════════════════════════════════════════════════════════
# Integrated Gradients
# ═══════════════════════════════════════════════════════════════════


def compute_integrated_gradients(
    ctx: DiagContext,
    query_emb: torch.Tensor,
    sup_feat: torch.Tensor,
    sup_mask: torch.Tensor,
    dense_prompt: torch.Tensor,
    H: int, W: int,
    gt: np.ndarray,
    n_steps: int = 50,
) -> dict:
    """Compute Integrated Gradients for prompt channels.

    IG(ch_i) = (prompt_ch_i - baseline_ch_i)
               × Σ_k (∂loss/∂prompt_ch_i at α_k) / n_steps

    :param dense_prompt: [1, C, gh, gw] actual prompt.
    :param H, W: original tile dimensions.
    :param gt: [H, W] boolean GT mask.
    :param n_steps: number of interpolation steps.
    :return: dict with per-channel IG, per-spatial IG, and verification metrics.
    """
    model = ctx.model
    device = ctx.device
    C = dense_prompt.shape[1]
    baseline = torch.zeros_like(dense_prompt)

    # GT at 256x256 for loss
    gt_t = torch.from_numpy(gt.astype(np.float32)).to(device)
    gt_256 = F.interpolate(
        gt_t.unsqueeze(0).unsqueeze(0), (256, 256), mode="area"
    )[0, 0]

    support_proto = model._compute_support_prototype(sup_feat, sup_mask) \
        if model.bypass_head is None else None

    # Per-channel IG accumulator
    ch_ig = torch.zeros(C, device=device)
    # Spatial IG accumulator (pixel-level in 64x64 grid)
    spatial_ig = torch.zeros(dense_prompt.shape[2], dense_prompt.shape[3], device=device)

    ig_path = []

    for k in range(n_steps):
        alpha = k / (n_steps - 1) if n_steps > 1 else 1.0
        # Interpolated prompt
        interp = baseline + alpha * (dense_prompt - baseline)  # [1, C, gh, gw]
        interp = interp.detach().requires_grad_(True)

        # Decode → loss
        if model.bypass_head is not None:
            low_res = model.bypass_head(interp)
        else:
            sparse_token = interp.mean(dim=(2, 3))
            # Temporarily disable cat injection for clean gradient path
            saved_cat = model.sam_decoder._category_enabled
            model.sam_decoder._category_enabled = False
            low_res, _ = model.sam_decoder(
                query_emb, sparse_token, interp,
                support_prototype=support_proto,
            )
            model.sam_decoder._category_enabled = saved_cat

        # Loss = BCE + Dice (same as decoder_inversion)
        fg = low_res[0, 0]
        bce = F.binary_cross_entropy_with_logits(fg, gt_256)
        prob = fg.sigmoid()
        inter = (prob * gt_256).sum()
        dice = 1.0 - (2 * inter + 1) / (prob.sum() + gt_256.sum() + 1)
        loss = bce + dice

        # Gradients w.r.t. prompt
        grads = torch.autograd.grad(loss, interp, retain_graph=False)[0]  # [1, C, gh, gw]

        with torch.no_grad():
            # Per-channel IG: sum over spatial, add to accumulator
            step_ch_grad = grads[0].mean(dim=(1, 2))  # [C]
            ch_ig += step_ch_grad / n_steps

            # Per-spatial IG: sum over channel, add to accumulator
            step_sp_grad = grads[0].abs().mean(dim=0)  # [gh, gw]
            spatial_ig += step_sp_grad / n_steps

            if k % 10 == 0 or k == n_steps - 1:
                ig_path.append({"alpha": round(alpha, 3), "loss": float(loss)})

        # Clean up to avoid graph accumulation
        del loss, low_res, grads, interp

    # Final IG = (prompt - baseline) × mean gradient
    with torch.no_grad():
        # Per-channel IG score
        ig_ch = (dense_prompt[0] - baseline[0]).mean(dim=(1, 2)) * ch_ig  # [C]

        # Channel ranking
        ch_scores = ig_ch.cpu().numpy()
        ranked = np.argsort(-np.abs(ch_scores))  # descending by |IG|

    # ── IG Top-K verification ──
    # Test: zero out low-IG channels, keep only Top-K (K ∈ {4, 8, 16, 32, 64, 128})
    topk_ious = {}
    full_iou = _quick_iou(ctx, dense_prompt, query_emb, sup_feat, sup_mask, gt, H, W)

    for k in [4, 8, 16, 32, 64, 128, 256]:
        keep = set(ranked[:k].tolist())
        mask_t = torch.zeros(C, device=device)
        mask_t[list(keep)] = 1.0
        truncated = dense_prompt * mask_t.view(1, C, 1, 1)
        topk_ious[f"ig_top{k}"] = _quick_iou(ctx, truncated, query_emb, sup_feat, sup_mask, gt, H, W)

    # Also test bottom-K for comparison
    for k in [4, 8, 16, 32]:
        keep = set(ranked[-k:].tolist())
        mask_t = torch.zeros(C, device=device)
        mask_t[list(keep)] = 1.0
        truncated = dense_prompt * mask_t.view(1, C, 1, 1)
        topk_ious[f"ig_bottom{k}"] = _quick_iou(ctx, truncated, query_emb, sup_feat, sup_mask, gt, H, W)

    return {
        "ig_ch": ch_scores.tolist(),          # [256] per-channel IG scores
        "ig_ch_abs": np.abs(ch_scores).tolist(),
        "ig_spatial": spatial_ig.cpu().numpy().tolist(),  # [64, 64]
        "ranked_channels": ranked.tolist(),    # channel indices ranked by |IG|
        "topk_ious": topk_ious,
        "full_iou": full_iou,
        "ig_path": ig_path,
        "n_pos_ig": int((ch_scores > 1e-5).sum()),
        "n_neg_ig": int((ch_scores < -1e-5).sum()),
    }


@torch.no_grad()
def _quick_iou(
    ctx: DiagContext,
    prompt: torch.Tensor,
    query_emb: torch.Tensor,
    sup_feat: torch.Tensor,
    sup_mask: torch.Tensor,
    gt: np.ndarray,
    H: int, W: int,
) -> float:
    """Quick IoU evaluation through the decoder."""
    model = ctx.model
    if model.bypass_head is not None:
        low_res = model.bypass_head(prompt)
    else:
        sparse_token = prompt.mean(dim=(2, 3))
        support_proto = model._compute_support_prototype(sup_feat, sup_mask)
        saved_cat = model.sam_decoder._category_enabled
        model.sam_decoder._category_enabled = False
        low_res, _ = model.sam_decoder(query_emb, sparse_token, prompt,
                                        support_prototype=support_proto)
        model.sam_decoder._category_enabled = saved_cat

    pred_logits = F.interpolate(
        low_res.float(), size=(H, W), mode="bilinear", align_corners=False,
    )[0, 0]
    vals = pred_logits.cpu()
    if vals.min() >= 0 and vals.max() <= 1:
        pred = vals > 0.5
    else:
        pred = vals.sigmoid() > 0.5
    pred_np = pred.numpy()
    inter = (pred_np & gt).sum()
    union = (pred_np | gt).sum()
    return float(inter / union) if union > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════
# Correlation Channel Ranking (for comparison)
# ═══════════════════════════════════════════════════════════════════


def compute_channel_gt_correlation(
    dense_prompt: torch.Tensor, gt: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Compute per-channel Pearson r with GT (for comparison with IG)."""
    C = dense_prompt.shape[1]
    gt_t = torch.from_numpy(gt.astype(np.float32))
    gh, gw = dense_prompt.shape[2], dense_prompt.shape[3]
    gt_rsz = F.interpolate(gt_t.unsqueeze(0).unsqueeze(0), (gh, gw), mode="area")[0, 0].numpy()
    gt_f = gt_rsz.ravel()

    cors = np.zeros(C)
    s_g = gt_f.std()
    for c in range(C):
        ch_f = dense_prompt[0, c].cpu().numpy().ravel()
        s_c = ch_f.std()
        cors[c] = float(np.corrcoef(ch_f, gt_f)[0, 1]) if s_c > 1e-8 and s_g > 1e-8 else 0.0

    ranked = np.argsort(-np.abs(cors))

    # Spearman correlation between IG and Pearson
    from scipy import stats as scipy_stats
    spearman_r, spearman_p = scipy_stats.spearmanr(
        np.abs(cors), np.abs(dense_prompt[0].mean(dim=(1, 2)).cpu().numpy())
    )

    return cors, {
        "pearson_r": cors.tolist(),
        "pearson_ranked": ranked.tolist(),
        "n_pos_ch": int((cors > 0.1).sum()),
        "n_neg_ch": int((cors < -0.1).sum()),
        "best_ch_r": float(cors.max()),
        "worst_ch_r": float(cors.min()),
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Decoder Integrated Gradients — Causal Channel Attribution"
    )
    add_common_args(parser)
    parser.add_argument("--ig-steps", type=int, default=50,
                       help="Number of IG interpolation steps")
    args = parser.parse_args()

    ctx = build_diag_context(args, require_adapter=True, split="val")
    build_support_cache(ctx)
    select_tiles(ctx, num_tiles=args.num_tiles)

    per_tile = []
    all_ig_ch = []  # for aggregating IG scores across tiles

    for tile_idx, present_classes in tqdm(ctx.selected_tiles, desc="integrated gradients"):
        sample = ctx.dataset[tile_idx]
        H, W = sample["image"].shape[1], sample["image"].shape[2]

        # Embed query
        xx, _ = preprocess_image(sample["image"])
        query_emb = ctx.backbone(xx.unsqueeze(0).to(ctx.device))["image_embedding"]
        if ctx.adapter is not None:
            query_emb = ctx.adapter(query_emb)

        # Pick main class
        main_cls = max(present_classes, key=lambda c:
                       ctx.dataset.get_class_mask(tile_idx, c).sum())
        gt_main = ctx.dataset.get_class_mask(tile_idx, main_cls).numpy().astype(bool)

        sup_data = ctx.support_cache.get(main_cls)
        if sup_data is None:
            continue
        sup_feat, sup_mask = sup_data

        # Extract dense prompt
        dense_prompt, semantic_prior, geometric_prior = extract_dense_prompt(
            ctx, query_emb, sup_feat, sup_mask
        )

        # Compute IG
        ig_result = compute_integrated_gradients(
            ctx, query_emb, sup_feat, sup_mask, dense_prompt,
            H, W, gt_main, n_steps=args.ig_steps,
        )

        # Compute channel-GT Pearson for comparison
        pearson_cors, pearson_info = compute_channel_gt_correlation(
            dense_prompt, gt_main
        )

        # Spearman between IG and Pearson
        from scipy import stats as scipy_stats
        ig_abs = np.abs(np.array(ig_result["ig_ch"]))
        pearson_abs = np.abs(pearson_cors)
        spearman_r, _ = scipy_stats.spearmanr(ig_abs, pearson_abs)

        entry = {
            "tile_idx": tile_idx,
            "class_id": main_cls,
            "class_name": ISAID5I_CATEGORIES.get(main_cls, f"cls{main_cls}"),
            "full_iou": ig_result["full_iou"],
            "n_pos_ig": ig_result["n_pos_ig"],
            "n_neg_ig": ig_result["n_neg_ig"],
            "topk_ious": ig_result["topk_ious"],
            "spearman_ig_vs_pearson": round(float(spearman_r), 4),
            "n_pos_pearson": pearson_info["n_pos_ch"],
            "n_neg_pearson": pearson_info["n_neg_ch"],
            "best_ch_pearson_r": pearson_info["best_ch_r"],
        }
        per_tile.append(entry)
        all_ig_ch.append(np.array(ig_result["ig_ch"]))

    # ── Aggregate ──
    summary = {"n_tiles": len(per_tile)}
    summary["metrics"] = aggregate_metrics(per_tile)

    # Aggregate IG scores across tiles (mean absolute IG per channel)
    if all_ig_ch:
        ig_matrix = np.stack(all_ig_ch, axis=0)  # [N, 256]
        mean_ig = np.abs(ig_matrix).mean(axis=0)
        summary["mean_abs_ig_per_channel"] = {
            "mean": float(mean_ig.mean()),
            "std": float(mean_ig.std()),
            "top10_channels": np.argsort(-mean_ig)[:10].tolist(),
            "bottom10_channels": np.argsort(mean_ig)[:10].tolist(),
        }

    # ── Save ──
    s_path, p_path = save_results(ctx.out_dir, summary, per_tile)

    # ── Print report ──
    print_header("Decoder Integrated Gradients — Causal Channel Attribution")
    if "metrics" in summary:
        print("\n  ┌─ IoU Summary")
        print_metric_table({k: v for k, v in summary["metrics"].items()
                           if "iou" in k.lower()})

        print("\n  ┌─ IG Statistics")
        print_metric_table({k: v for k, v in summary["metrics"].items()
                           if "ig" in k.lower() or "spearman" in k.lower()})

        print("\n  ┌─ IG Top-K Verification (mean IoU)")
        for key in ["ig_top4", "ig_top8", "ig_top16", "ig_top32", "ig_top64", "ig_top128",
                     "ig_bottom4", "ig_bottom8", "ig_bottom16", "ig_bottom32"]:
            if key in summary["metrics"]:
                m = summary["metrics"][key]
                print(f"    {key:<20s} IoU={m['mean']:.4f} ± {m['std']:.4f}")

    if "mean_abs_ig_per_channel" in summary:
        mig = summary["mean_abs_ig_per_channel"]
        print(f"\n  ┌─ Per-Channel Mean |IG|: {mig['mean']:.6f} ± {mig['std']:.6f}")
        print(f"    Top-10 channels (highest |IG|): {mig['top10_channels']}")

    print(f"\n[Summary] {s_path}")
    print(f"[Per-tile] {p_path}")
    print("[Done]")


if __name__ == "__main__":
    main()
