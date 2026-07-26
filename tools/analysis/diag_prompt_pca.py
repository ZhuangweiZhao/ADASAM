"""
Prompt PCA — Effective Dimensionality of Dense Prompt
======================================================

核心问题: 256 个 channel 中有多少维是真正被 Decoder 利用的？

方法:
  1. 收集 N 个 tile 的 dense_prompt [N, 256, 64, 64]
  2. 对 channel 维度做 PCA: reshape → [N, 256] (spatial mean)
  3. 计算各 PC 解释的方差比例, 找到有效维度
  4. Truncation 实验: 投影到 top-K PC 后重建, 测试 mIoU
     K ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256}
  5. 对比三种 prompt:
     - SPG-only: semantic_prior 直接作为 prompt
     - Current: PromptFusion 输出
     - Geo-only: geometric_prior 直接作为 prompt

输出:
  - Cumulative variance explained vs K
  - mIoU vs K (性能饱和曲线)
  - 90%/95%/99% variance 对应的 K 值

用法:
    python tools/analysis/diag_prompt_pca.py \
        --stage2-ckpt runs/stage2_fold1_k5_seed42/best_model.pt \
        --data-root data/iSAID-5i --fold 1 --mode novel --k-shot 5 \
        --num-tiles 50 --pca-tiles 50
"""

from __future__ import annotations

import argparse
import json
import random
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
    eval_iou,
    print_header,
    print_metric_table,
    aggregate_metrics,
    save_results,
)


# ═══════════════════════════════════════════════════════════════════
# Prompt Collection
# ═══════════════════════════════════════════════════════════════════


@torch.no_grad()
def collect_prompts(
    ctx: DiagContext,
    num_tiles: int = 50,
) -> dict[str, np.ndarray]:
    """Collect dense prompts from multiple tiles for PCA.

    Also collects SPG-only and Geo-only variants.

    :return: {"current": [N, 256], "spg_only": [N, 256], "geo_only": [N, 256]}
    """
    prompts = {"current": [], "spg_only": [], "geo_only": []}
    meta = {"tile_indices": [], "class_ids": [], "ious": []}

    tiles_used = ctx.selected_tiles if ctx.selected_tiles else select_tiles(ctx, num_tiles=num_tiles)
    tiles_used = tiles_used[:num_tiles]

    for tile_idx, present_classes in tqdm(tiles_used, desc="collect prompts"):
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

        # Current (PromptFusion)
        dense_prompt, semantic_prior, geometric_prior = extract_dense_prompt(
            ctx, query_emb, sup_feat, sup_mask
        )

        # Spatial-mean pooling for PCA
        prompts["current"].append(dense_prompt[0].mean(dim=(1, 2)).cpu().numpy())  # [256]
        prompts["spg_only"].append(semantic_prior[0].mean(dim=(1, 2)).cpu().numpy())
        if geometric_prior is not None:
            prompts["geo_only"].append(geometric_prior[0].mean(dim=(1, 2)).cpu().numpy())

        iou_val = eval_iou(ctx.model, dense_prompt, query_emb, sup_feat, sup_mask, gt, H, W)
        meta["ious"].append(iou_val)
        meta["tile_indices"].append(tile_idx)
        meta["class_ids"].append(main_cls)

    return {k: np.stack(v, axis=0) if v else np.array([]) for k, v in prompts.items()}, meta


# ═══════════════════════════════════════════════════════════════════
# PCA Analysis
# ═══════════════════════════════════════════════════════════════════


def pca_analysis(prompt_matrix: np.ndarray) -> dict:
    """Run PCA on prompt collection [N, 256].

    :return: dict with eigenvalues, explained variance, PCs.
    """
    N, C = prompt_matrix.shape
    # Center
    mean = prompt_matrix.mean(axis=0, keepdims=True)
    centered = prompt_matrix - mean

    # SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # S: singular values [min(N, C)]
    # Vt: [min(N, C), C] — principal components

    eigenvalues = S ** 2 / (N - 1) if N > 1 else S ** 2
    total_var = eigenvalues.sum()
    explained = eigenvalues / total_var if total_var > 0 else np.zeros_like(eigenvalues)
    cumsum = np.cumsum(explained)

    # Effective dimensions at various thresholds
    k_90 = int(np.searchsorted(cumsum, 0.90) + 1)
    k_95 = int(np.searchsorted(cumsum, 0.95) + 1)
    k_99 = int(np.searchsorted(cumsum, 0.99) + 1)

    return {
        "N": N,
        "C": C,
        "eigenvalues": eigenvalues.tolist(),
        "explained_variance": explained.tolist(),
        "cumulative_variance": cumsum.tolist(),
        "k_90": k_90,
        "k_95": k_95,
        "k_99": k_99,
        "total_variance": float(total_var),
        "pc_matrix": Vt.tolist(),  # [min(N,C), 256]
        "mean": mean[0].tolist(),
    }


# ═══════════════════════════════════════════════════════════════════
# PCA Truncation Experiment
# ═══════════════════════════════════════════════════════════════════


@torch.no_grad()
def pca_truncation_test(
    ctx: DiagContext,
    prompt: torch.Tensor,
    pca_result: dict,
    query_emb: torch.Tensor,
    sup_feat: torch.Tensor,
    sup_mask: torch.Tensor,
    gt: np.ndarray,
    H: int, W: int,
    K_list: list[int] = None,
) -> dict:
    """Project prompt onto top-K PCs and test mIoU.

    :return: {f"pca_k{K}": iou_float, ...}
    """
    if K_list is None:
        K_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    C = prompt.shape[1]
    Vt = np.array(pca_result["pc_matrix"])  # [min(N,C), C]
    mean = np.array(pca_result["mean"])      # [C]

    # Flatten prompt to channel-mean representation [C]
    p_flat = prompt[0].mean(dim=(1, 2)).cpu().numpy()  # [C]
    centered = p_flat - mean

    truncation_ious = {}
    for K in K_list:
        if K >= Vt.shape[0]:
            K_eff = Vt.shape[0]
        else:
            K_eff = K

        # Project onto top-K PCs and reconstruct
        coeffs = centered @ Vt[:K_eff].T  # [K]
        reconstructed = mean + coeffs @ Vt[:K_eff]  # [C]

        # Create truncated prompt: replace channel means with PCA-reconstructed values
        truncated = prompt.clone()
        # Scale each channel to match the PCA-reconstructed mean
        ch_mean_orig = prompt[0].mean(dim=(1, 2))  # [C]
        ch_std_orig = prompt[0].std(dim=(1, 2))    # [C]

        for c in range(C):
            if ch_std_orig[c] > 1e-8:
                # Rescale: keep spatial pattern, shift mean to PCA value
                truncated[0, c] = prompt[0, c] - ch_mean_orig[c] + reconstructed[c]

        iou_val = eval_iou(ctx.model, truncated, query_emb, sup_feat, sup_mask, gt, H, W)
        truncation_ious[f"pca_k{K}"] = iou_val

    # Also record full IoU
    full_iou = eval_iou(ctx.model, prompt, query_emb, sup_feat, sup_mask, gt, H, W)
    truncation_ious["pca_k_full"] = full_iou

    return truncation_ious


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Prompt PCA — Effective Dimensionality Analysis"
    )
    add_common_args(parser)
    parser.add_argument("--pca-tiles", type=int, default=50,
                       help="Number of tiles for PCA collection")
    parser.add_argument("--tiles4eval", type=int, default=20,
                       help="Tiles for truncation evaluation (subset of PCA tiles)")
    args = parser.parse_args()

    ctx = build_diag_context(args, require_adapter=True, split="val")
    build_support_cache(ctx)

    # ── Phase 1: Collect prompts ──
    print_header("Phase 1: Collecting Prompts for PCA")
    select_tiles(ctx, num_tiles=args.pca_tiles)
    prompt_dict, meta = collect_prompts(ctx, num_tiles=args.pca_tiles)

    # ── Phase 2: PCA Analysis ──
    print_header("Phase 2: PCA Analysis")
    pca_results = {}
    for prompt_type in ["current", "spg_only", "geo_only"]:
        if prompt_dict[prompt_type].size == 0:
            continue
        pca = pca_analysis(prompt_dict[prompt_type])
        pca_results[prompt_type] = pca
        print(f"\n  [{prompt_type}]")
        print(f"    N={pca['N']}, total_variance={pca['total_variance']:.4f}")
        print(f"    K(90%)={pca['k_90']}, K(95%)={pca['k_95']}, K(99%)={pca['k_99']}")
        print(f"    Top-5 PCs explain: {pca['cumulative_variance'][4]:.1%}")

    # ── Phase 3: Truncation Experiment ──
    print_header("Phase 3: PCA Truncation Experiment")
    truncation_results = []
    K_list = [1, 2, 4, 8, 16, 32, 64, 128]

    # Use first `tiles4eval` tiles for truncation test
    eval_tiles = ctx.selected_tiles[:args.tiles4eval]
    tiles_tested = 0

    for tile_idx, present_classes in tqdm(eval_tiles, desc="truncation"):
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

        dense_prompt, _, _ = extract_dense_prompt(ctx, query_emb, sup_feat, sup_mask)

        if "current" in pca_results and pca_results["current"]["pc_matrix"]:
            ious = pca_truncation_test(
                ctx, dense_prompt, pca_results["current"],
                query_emb, sup_feat, sup_mask, gt, H, W, K_list=K_list,
            )
            ious["tile_idx"] = tile_idx
            ious["class_id"] = main_cls
            ious["class_name"] = ISAID5I_CATEGORIES.get(main_cls, f"cls{main_cls}")
            truncation_results.append(ious)
            tiles_tested += 1

    # ── Aggregate ──
    summary = {
        "pca": pca_results,
        "n_pca_tiles": meta.get("ious", []) and len(meta["ious"]) or 0,
    }

    if truncation_results:
        summary["truncation"] = aggregate_metrics(truncation_results)

    s_path, p_path = save_results(ctx.out_dir, summary, truncation_results)

    # ── Print Truncation Report ──
    if "truncation" in summary:
        print("\n  ┌─ PCA Truncation: mIoU vs K (Current Prompt)")
        for K in K_list:
            key = f"pca_k{K}"
            if key in summary["truncation"]:
                m = summary["truncation"][key]
                print(f"    K={K:>3d}  IoU={m['mean']:.4f} ± {m['std']:.4f}  "
                      f"(min={m['min']:.4f}, max={m['max']:.4f})")
        if "pca_k_full" in summary["truncation"]:
            m = summary["truncation"]["pca_k_full"]
            print(f"    K=full  IoU={m['mean']:.4f} ± {m['std']:.4f}")

    print(f"\n[Summary] {s_path}")
    print(f"[Per-tile] {p_path}")
    print("[Done]")


if __name__ == "__main__":
    main()
