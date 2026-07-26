"""
Prompt Gradient Health — 统计 PromptFusion 输出各 channel 的梯度分布
=====================================================================

核心问题: 是否 90% channel 的梯度 ≈ 0（梯度死亡），导致 PromptFusion 无学习压力？

这是因果验证 Priority 2: 如果梯度的确大面积死亡，则 Rank Collapse 是梯度问题的
**结果**而非**原因**。

测量每个 channel:
  - gradient norm (总梯度大小)
  - gradient mean (平均梯度)
  - gradient variance (空间梯度变化)
  - gradient entropy (梯度集中度: 低=集中少数像素, 高=均匀分布)
  - gradient alive fraction (|grad| > 1e-8 的比例)

跨 tile 统计:
  - "dead channels" (|grad_norm| < 1e-6 的 channel 比例)
  - per-channel gradient norm 分布 (mean/std/min/max)
  - gradient Gini (梯度是否集中在少数 channel)

用法:
    python tools/debug/diag_prompt_gradient_health.py \
        --stage2-ckpt runs/stage2_fold1_k5_seed42/best_model.pt \
        --data-root data/iSAID-5i --fold 1 --mode novel --k-shot 5 \
        --num-tiles 30
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

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
    save_results,
)


# ═══════════════════════════════════════════════════════════════════
# Gradient Health Analysis
# ═══════════════════════════════════════════════════════════════════


def compute_gradient_health(
    ctx: DiagContext,
    query_emb: torch.Tensor,
    sup_feat: torch.Tensor,
    sup_mask: torch.Tensor,
    dense_prompt: torch.Tensor,
    H: int, W: int,
    gt: np.ndarray,
) -> dict:
    """Compute per-channel gradient statistics for the dense prompt.

    Measures gradient health: are prompt channels receiving usable gradients?

    :return: dict with per-channel gradient stats.
    """
    model = ctx.model
    device = ctx.device
    C = dense_prompt.shape[1]

    # Detach inputs to isolate gradient path to prompt only
    query_emb = query_emb.detach()
    sup_feat_d = sup_feat.detach()
    sup_mask_d = sup_mask.detach()

    # GT at 256×256 for loss
    gt_t = torch.from_numpy(gt.astype(np.float32)).to(device)
    gt_256 = F.interpolate(
        gt_t.unsqueeze(0).unsqueeze(0), (256, 256), mode="area"
    )[0, 0]

    # Make prompt require grad
    prompt = dense_prompt.detach().clone().requires_grad_(True)

    # Decode
    support_proto = None
    if model.bypass_head is None:
        support_proto = model._compute_support_prototype(sup_feat_d, sup_mask_d)
        saved_cat = model.sam_decoder._category_enabled
        model.sam_decoder._category_enabled = False

    if model.bypass_head is not None:
        low_res = model.bypass_head(prompt)
    else:
        sparse_token = prompt.mean(dim=(2, 3))
        low_res, _ = model.sam_decoder(
            query_emb, sparse_token, prompt,
            support_prototype=support_proto,
        )

    if model.bypass_head is None:
        model.sam_decoder._category_enabled = saved_cat

    # Loss
    fg = low_res[0, 0]
    bce = F.binary_cross_entropy_with_logits(fg, gt_256)
    prob = fg.sigmoid()
    inter = (prob * gt_256).sum()
    dice = 1.0 - (2 * inter + 1) / (prob.sum() + gt_256.sum() + 1)
    loss = bce + dice

    # Gradient of loss w.r.t. prompt
    grad = torch.autograd.grad(loss, prompt, retain_graph=False)[0]  # [1, C, gh, gw]

    with torch.no_grad():
        grad_ch = grad[0]  # [C, gh, gw]
        gh, gw = grad_ch.shape[1], grad_ch.shape[2]

        # Per-channel statistics
        ch_grad_norm = grad_ch.reshape(C, -1).norm(dim=1).cpu().numpy()       # [C]
        ch_grad_mean = grad_ch.reshape(C, -1).mean(dim=1).cpu().numpy()       # [C]
        ch_grad_var  = grad_ch.reshape(C, -1).var(dim=1).cpu().numpy()        # [C]

        # Per-channel gradient entropy (treat |grad| as spatial distribution)
        ch_entropy = np.zeros(C)
        for c in range(C):
            g_abs = grad_ch[c].abs().reshape(-1)
            g_sum = g_abs.sum()
            if g_sum > 1e-10:
                g_prob = g_abs / g_sum
                ch_entropy[c] = float(
                    -(g_prob * torch.log(g_prob + 1e-10)).sum() / np.log(gh * gw)
                )

        # Per-channel gradient alive fraction (|grad| > 1e-8)
        ch_alive = np.zeros(C)
        for c in range(C):
            ch_alive[c] = float((grad_ch[c].abs() > 1e-8).float().mean())

        # Channel gradient Gini (how concentrated are gradients across channels?)
        sorted_norms = np.sort(np.abs(ch_grad_norm))
        n = len(sorted_norms)
        gini = 1.0 - 2.0 * np.sum(sorted_norms * np.arange(1, n + 1)) / (n * sorted_norms.sum() + 1e-10)

        # Dead channels: |grad_norm| < threshold
        dead_1e6 = int((np.abs(ch_grad_norm) < 1e-6).sum())
        dead_1e8 = int((np.abs(ch_grad_norm) < 1e-8).sum())
        active_1e4 = int((np.abs(ch_grad_norm) > 1e-4).sum())

        # Top channels by gradient norm
        top10 = np.argsort(-np.abs(ch_grad_norm))[:10].tolist()
        bottom10 = np.argsort(np.abs(ch_grad_norm))[:10].tolist()

        # Compare with prompt channel energy
        ch_energy = dense_prompt[0].pow(2).reshape(C, -1).mean(dim=1).cpu().numpy()
        # Spearman correlation: gradient norm vs channel energy
        from scipy import stats as scipy_stats
        spearman_grad_energy, _ = scipy_stats.spearmanr(np.abs(ch_grad_norm), ch_energy)

    # Save loss before cleanup
    loss_val = float(loss)

    del loss, low_res, grad, prompt

    return {
        "ch_grad_norm": ch_grad_norm.tolist(),
        "ch_grad_mean": ch_grad_mean.tolist(),
        "ch_grad_var": ch_grad_var.tolist(),
        "ch_grad_entropy": ch_entropy.tolist(),
        "ch_alive_fraction": ch_alive.tolist(),
        "grad_gini": float(gini),
        "n_dead_1e6": dead_1e6,
        "n_dead_1e8": dead_1e8,
        "n_active_1e4": active_1e4,
        "mean_grad_norm": float(np.abs(ch_grad_norm).mean()),
        "median_grad_norm": float(np.median(np.abs(ch_grad_norm))),
        "max_grad_norm": float(np.abs(ch_grad_norm).max()),
        "top10_by_grad": top10,
        "bottom10_by_grad": bottom10,
        "spearman_grad_vs_energy": float(spearman_grad_energy),
        "loss": loss_val,
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Prompt Gradient Health — per-channel gradient statistics"
    )
    add_common_args(parser)
    parser.set_defaults(num_tiles=30)  # override base default of 20
    parser.add_argument("--dead-threshold", type=float, default=1e-6,
                       help="Gradient norm below which a channel is considered 'dead'")
    args = parser.parse_args()

    ctx = build_diag_context(args, require_adapter=True, split="val")
    build_support_cache(ctx)
    select_tiles(ctx, num_tiles=args.num_tiles)

    per_tile = []
    # Aggregate per-channel gradient norms across tiles
    all_grad_norms = []  # [N_tiles, 256]

    for tile_idx, present_classes in tqdm(ctx.selected_tiles, desc="gradient health"):
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

        # Extract dense prompt
        dense_prompt, _, _ = extract_dense_prompt(ctx, query_emb, sup_feat, sup_mask)

        # Compute gradient health
        health = compute_gradient_health(
            ctx, query_emb, sup_feat, sup_mask, dense_prompt, H, W, gt,
        )

        health["tile_idx"] = tile_idx
        health["class_id"] = main_cls
        health["class_name"] = ISAID5I_CATEGORIES.get(main_cls, f"cls{main_cls}")
        per_tile.append(health)
        all_grad_norms.append(np.abs(np.array(health["ch_grad_norm"])))

    # ── Aggregate across tiles ──
    summary: dict = {"n_tiles": len(per_tile)}

    # Aggregate scalars
    scalar_keys = [
        "grad_gini", "n_dead_1e6", "n_dead_1e8", "n_active_1e4",
        "mean_grad_norm", "median_grad_norm", "max_grad_norm",
        "spearman_grad_vs_energy", "loss",
    ]
    agg = {}
    for key in scalar_keys:
        vals = [d[key] for d in per_tile if key in d and d[key] is not None]
        if vals:
            agg[key] = {
                "mean": round(float(np.mean(vals)), 6),
                "median": round(float(np.median(vals)), 6),
                "std": round(float(np.std(vals)), 6),
                "min": round(float(np.min(vals)), 6),
                "max": round(float(np.max(vals)), 6),
            }
    summary["aggregates"] = agg

    # Aggregate per-channel gradient norms across tiles
    if all_grad_norms:
        gn_matrix = np.stack(all_grad_norms, axis=0)  # [N, 256]
        mean_per_ch = gn_matrix.mean(axis=0)  # [256]

        # "Dead channels across tiles": channels consistently dead
        dead_consistently = int((mean_per_ch < args.dead_threshold).sum())
        summary["per_channel_cross_tile"] = {
            "mean_grad_norm_per_ch": mean_per_ch.tolist(),
            "std_grad_norm_per_ch": gn_matrix.std(axis=0).tolist(),
            "mean_of_mean": float(mean_per_ch.mean()),
            "median_of_mean": float(np.median(mean_per_ch)),
            "n_consistently_dead": dead_consistently,
            "dead_fraction": round(dead_consistently / 256, 4),
            "top10_channels_cross_tile": np.argsort(-mean_per_ch)[:10].tolist(),
        }

        # Gradient norm distribution summary
        all_norms_flat = gn_matrix.ravel()
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        summary["grad_norm_distribution"] = {
            f"p{p}": round(float(np.percentile(all_norms_flat, p)), 8)
            for p in percentiles
        }

    s_path, p_path = save_results(ctx.out_dir, summary, per_tile)

    # ── Print report ──
    print_header("Prompt Gradient Health — Per-Channel Gradient Statistics")

    if "aggregates" in summary:
        a = summary["aggregates"]
        print("\n  ┌─ Gradient Norm Distribution (across tiles)")
        for key, label in [
            ("mean_grad_norm", "Mean |grad| per channel"),
            ("median_grad_norm", "Median |grad| per channel"),
            ("max_grad_norm", "Max |grad| per channel"),
        ]:
            if key in a:
                print(f"    {label:<35s} {a[key]['mean']:.2e} ± {a[key]['std']:.2e}")

        print("\n  ┌─ Channel Gradient Concentration")
        for key, label in [
            ("grad_gini", "Gradient Gini (0=equal, 1=concentrated)"),
        ]:
            if key in a:
                print(f"    {label:<35s} {a[key]['mean']:.4f}")

        print("\n  ┌─ Dead / Active Channels (|grad| thresholds)")
        for key, label in [
            ("n_dead_1e8", "Dead (|g| < 1e-8)"),
            ("n_dead_1e6", "Dead (|g| < 1e-6)"),
            ("n_active_1e4", "Active (|g| > 1e-4)"),
        ]:
            if key in a:
                print(f"    {label:<35s} {a[key]['mean']:.1f} / 256  "
                      f"({100*a[key]['mean']/256:.1f}%)")

        print("\n  ┌─ Gradient vs Channel Energy")
        if "spearman_grad_vs_energy" in a:
            print(f"    Spearman(grad_norm, ch_energy) = {a['spearman_grad_vs_energy']['mean']:.4f}")

    if "per_channel_cross_tile" in summary:
        pcc = summary["per_channel_cross_tile"]
        print(f"\n  ┌─ Cross-Tile Per-Channel Consistency")
        print(f"    Consistently dead channels: {pcc['n_consistently_dead']} / 256 "
              f"({pcc['dead_fraction']:.1%})")
        print(f"    Mean-of-mean |grad|: {pcc['mean_of_mean']:.2e}")
        print(f"    Top-10 channels (highest |grad|): {pcc['top10_channels_cross_tile']}")

    if "grad_norm_distribution" in summary:
        gd = summary["grad_norm_distribution"]
        print(f"\n  ┌─ Gradient Norm Percentiles (all channels × all tiles)")
        print(f"    p1={gd['p1']:.2e}  p5={gd['p5']:.2e}  p10={gd['p10']:.2e}  "
              f"p50={gd['p50']:.2e}  p90={gd['p90']:.2e}  p99={gd['p99']:.2e}")

    print(f"\n[Summary] {s_path}")
    print(f"[Per-tile] {p_path}")
    print("[Done]")


if __name__ == "__main__":
    main()
