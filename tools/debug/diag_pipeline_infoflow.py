"""
Prompt Pipeline Information Flow — 完整信息流诊断
==================================================

最后一个因果实验: PromptFusion 输入 (concat(Geo, SPG)) 的有效维度是多少？

如果 Concat rank ≈ 2  → 问题在上游 (SPG/Geo 高度冗余), PF 无罪
如果 Concat rank ≈ 10 → 问题在 PF (坍缩 10D → 2D), PF 是瓶颈

测量链路中每个节点的:
  - Effective rank (SVD, 90% variance threshold)
  - PCA explained variance curve
  - Channel energy Gini
  - Cross-node CCA (canonical correlation)
  - Channel correlation statistics (mean |off-diag|)
  - Per-tile IoU (through decoder)

用法:
    python tools/debug/diag_pipeline_infoflow.py \
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
    save_results,
    aggregate_metrics,
)


# ═══════════════════════════════════════════════════════════════════
# Node-level analysis
# ═══════════════════════════════════════════════════════════════════


def node_analysis(x: torch.Tensor, name: str) -> dict:
    """Analyze a single node in the pipeline.

    :param x: [1, C, H, W] tensor (prompt-like).
    :return: dict with rank, PCA, gini, correlation.
    """
    C = x.shape[1]
    flat = x[0].reshape(C, -1).float()  # [C, HW]

    # SVD → effective rank
    U, S, Vh = torch.linalg.svd(flat, full_matrices=False)
    s2 = S ** 2
    total_var = float(s2.sum())
    cumsum = torch.cumsum(s2, dim=0) / (total_var + 1e-10)
    k_50 = int((cumsum < 0.50).sum() + 1)
    k_90 = int((cumsum < 0.90).sum() + 1)
    k_95 = int((cumsum < 0.95).sum() + 1)
    k_99 = int((cumsum < 0.99).sum() + 1)

    # Cumulative variance curve (top-10)
    top10_var = [float(cumsum[min(i, len(cumsum) - 1)]) for i in range(10)]

    # Channel energy Gini
    energy = flat.pow(2).mean(dim=1)  # [C]
    sorted_e = torch.sort(energy)[0]
    n = len(sorted_e)
    idx = torch.arange(1, n + 1, device=sorted_e.device, dtype=torch.float32)
    gini = float(1 - 2 * torch.sum(sorted_e * idx) / (n * sorted_e.sum() + 1e-10) + (n + 1) / n)

    # Channel correlation (mean |off-diagonal|)
    # Sample 64 random channels for speed
    if C > 64:
        idx_sample = torch.randperm(C)[:64]
        flat_sample = flat[idx_sample]
    else:
        flat_sample = flat
    flat_n = F.normalize(flat_sample, dim=1)
    corr = flat_n @ flat_n.T
    mask = ~torch.eye(len(flat_sample), dtype=torch.bool, device=flat_sample.device)
    ch_correlation = float(corr[mask].abs().mean())

    # Top singular values
    top5_sv = S[:min(5, len(S))].cpu().tolist()

    return {
        "node": name,
        "C": C,
        "k_50": k_50, "k_90": k_90, "k_95": k_95, "k_99": k_99,
        "total_variance": total_var,
        "cumsum_top10": top10_var,
        "energy_gini": gini,
        "ch_correlation": ch_correlation,
        "top5_singular_values": top5_sv,
        "sv_ratio_s1_s2": float(S[0] / (S[1] + 1e-10)) if len(S) >= 2 else 0,
        "sv_ratio_s1_s5": float(S[0] / (S[min(4, len(S)-1)] + 1e-10)) if len(S) >= 5 else 0,
    }


# ═══════════════════════════════════════════════════════════════════
# CCA between two nodes
# ═══════════════════════════════════════════════════════════════════


def cca_analysis(x: torch.Tensor, y: torch.Tensor, n_components: int = 10) -> dict:
    """CCA between two nodes (spatial-mean pooled, or full spatial).

    :param x: [1, C1, H, W]
    :param y: [1, C2, H, W]
    :return: canonical correlations + mean/similarity metrics.
    """
    C1, C2 = x.shape[1], y.shape[1]
    # Spatial-mean pool → [1, C]
    x_p = x[0].mean(dim=(1, 2)).unsqueeze(0).float()  # [1, C1]
    y_p = y[0].mean(dim=(1, 2)).unsqueeze(0).float()  # [1, C2]

    # For CCA, need multiple samples. Use spatial positions as samples.
    x_sp = x[0].reshape(C1, -1).T.float()  # [HW, C1]
    y_sp = y[0].reshape(C2, -1).T.float()  # [HW, C2]

    # Center
    x_c = x_sp - x_sp.mean(dim=0, keepdim=True)
    y_c = y_sp - y_sp.mean(dim=0, keepdim=True)

    # Regularized CCA (Tikhonov)
    reg = 1e-4
    C_xx = (x_c.T @ x_c) / (x_c.shape[0] - 1) + reg * torch.eye(C1, device=x.device)
    C_yy = (y_c.T @ y_c) / (y_c.shape[0] - 1) + reg * torch.eye(C2, device=y.device)
    C_xy = (x_c.T @ y_c) / (x_c.shape[0] - 1)

    # Solve generalized eigenvalue problem via SVD of whitened cross-cov
    try:
        Lx = torch.linalg.cholesky(C_xx)
        Ly = torch.linalg.cholesky(C_yy)
        # Whitened cross-cov
        W = torch.linalg.solve(Lx, C_xy)
        W = torch.linalg.solve(Ly, W.T).T
        U_w, S_w, V_w = torch.linalg.svd(W, full_matrices=False)
        can_corrs = S_w[:n_components].cpu().tolist()
    except RuntimeError:
        # Cholesky failed → fallback to direct SVD of normalized matrices
        can_corrs = [0.0] * n_components

    # Also compute simpler metrics
    # Cosine similarity (only if same number of elements)
    x_flat = x[0].reshape(1, -1).float()
    y_flat = y[0].reshape(1, -1).float()
    if x_flat.shape[1] == y_flat.shape[1]:
        cos_global = float(F.cosine_similarity(x_flat, y_flat))
    else:
        # Use channel-mean vectors (both are spatial patches of same size)
        cos_global = float(F.cosine_similarity(
            x[0].mean(dim=(1, 2)).unsqueeze(0).float()[:, :min(C1, C2)],
            y[0].mean(dim=(1, 2)).unsqueeze(0).float()[:, :min(C1, C2)],
        ))

    # Subspace alignment: principal angles via SVD of X^T Y
    nx = F.normalize(x_c, dim=0)
    ny = F.normalize(y_c, dim=0)
    cross = nx.T @ ny  # [C1, C2]
    _, S_cross, _ = torch.linalg.svd(cross, full_matrices=False)
    principal_angles_cos = S_cross[:min(5, len(S_cross))].cpu().tolist()

    return {
        "canonical_correlations": can_corrs,
        "mean_cca": float(np.mean([c for c in can_corrs if not np.isnan(c)])),
        "cos_global": cos_global,
        "principal_angles_cos": principal_angles_cos,
        "mean_principal_angle_cos": float(np.mean(principal_angles_cos)) if principal_angles_cos else 0,
    }


# ═══════════════════════════════════════════════════════════════════
# Mutual Information (simple: based on PCA overlap)
# ═══════════════════════════════════════════════════════════════════


def subspace_overlap(x: torch.Tensor, y: torch.Tensor, top_k: int = 8) -> dict:
    """Measure overlap between top-K PCA subspaces of two nodes.

    :return: Grassmann distance, overlap ratio.
    """
    C1, C2 = x.shape[1], y.shape[1]
    x_f = x[0].reshape(C1, -1).float()  # [C1, HW]
    y_f = y[0].reshape(C2, -1).float()  # [C2, HW]

    # PCA bases
    _, _, Vx = torch.linalg.svd(x_f, full_matrices=False)  # Vx: [C1, HW]
    _, _, Vy = torch.linalg.svd(y_f, full_matrices=False)  # Vy: [C2, HW]

    k = min(top_k, Vx.shape[0], Vy.shape[0])
    Bx = Vx[:k]  # [k, HW]
    By = Vy[:k]  # [k, HW]

    # Projection matrices
    Px = Bx.T @ Bx  # [HW, HW] (rank k)
    Py = By.T @ By

    # Grassmann distance via Frobenius norm of difference
    grassmann_dist = float(torch.norm(Px - Py, p='fro') / np.sqrt(2 * k))

    # Overlap: how much of y's subspace is explained by x's PCs
    y_proj = By @ Bx.T  # [k, k] — projection coefficients
    overlap = float((y_proj ** 2).sum() / k)  # normalized

    return {
        "grassmann_dist_k8": grassmann_dist,
        "subspace_overlap_k8": overlap,
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Info Flow — complete prompt pathway diagnosis"
    )
    add_common_args(parser)
    parser.set_defaults(num_tiles=50)
    args = parser.parse_args()

    ctx = build_diag_context(args, require_adapter=True, split="val")
    build_support_cache(ctx)
    select_tiles(ctx, num_tiles=args.num_tiles)

    per_tile = []

    # Aggregate across tiles
    all_nodes = defaultdict(list)  # {node_name: [analysis_dict, ...]}
    all_ccas = defaultdict(list)   # {pair_name: [cca_dict, ...]}
    all_overlaps = defaultdict(list)

    for tile_idx, present_classes in tqdm(ctx.selected_tiles, desc="info flow"):
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

        # ── Extract all pipeline nodes ──
        model = ctx.model
        with torch.no_grad():
            support_memory = model.support_encoder(sup_feat, sup_mask)

            geometric_prior = None
            if model.geometric_prior is not None:
                geometric_prior = model.geometric_prior(query_emb, support_memory)

            dense_pe = model.sam_decoder.prompt_encoder.get_dense_pe()
            spg_out = model.spg(query_emb, support_memory, dense_pe)
            semantic_prior = spg_out.semantic_prior

            dense_prompt = None
            if model.prompt_fusion is not None and geometric_prior is not None:
                dense_prompt, _ = model.prompt_fusion(geometric_prior, semantic_prior)
            else:
                dense_prompt = semantic_prior

        # ── IoU for each node ──
        iou_geo = 0.0
        iou_spg = 0.0
        iou_pf = 0.0
        if geometric_prior is not None:
            iou_geo = eval_iou(model, geometric_prior, query_emb, sup_feat, sup_mask, gt, H, W)
        iou_spg = eval_iou(model, semantic_prior, query_emb, sup_feat, sup_mask, gt, H, W)
        iou_pf = eval_iou(model, dense_prompt, query_emb, sup_feat, sup_mask, gt, H, W)

        # ── Node analysis ──
        nodes = {}
        if geometric_prior is not None:
            nodes["Geo"] = node_analysis(geometric_prior, "Geo")
        nodes["SPG"] = node_analysis(semantic_prior, "SPG")
        nodes["PF"] = node_analysis(dense_prompt, "PF")

        # Concat(Geo, SPG) — the INPUT to PromptFusion
        if geometric_prior is not None:
            concat_input = torch.cat([geometric_prior, semantic_prior], dim=1)  # [1, 512, 64, 64]
            nodes["Concat"] = node_analysis(concat_input, "Concat(Geo,SPG)")

        # Add IoU to node stats
        if "Geo" in nodes:
            nodes["Geo"]["iou"] = iou_geo
        nodes["SPG"]["iou"] = iou_spg
        nodes["PF"]["iou"] = iou_pf
        if "Concat" in nodes:
            nodes["Concat"]["iou"] = iou_pf  # Concat feeds into PF, same IoU

        for name, n in nodes.items():
            n["tile_idx"] = tile_idx
            n["class_name"] = ISAID5I_CATEGORIES.get(main_cls, f"cls{main_cls}")
            all_nodes[name].append(n)

        # ── CCA between adjacent nodes ──
        ccas = {}
        if geometric_prior is not None:
            ccas["Geo↔SPG"] = cca_analysis(geometric_prior, semantic_prior)
            ccas["Geo↔Concat"] = cca_analysis(geometric_prior, concat_input)
            ccas["SPG↔Concat"] = cca_analysis(semantic_prior, concat_input)
            ccas["Concat↔PF"] = cca_analysis(concat_input, dense_prompt)
        for pair, c in ccas.items():
            all_ccas[pair].append(c)

        # ── Subspace overlap ──
        if geometric_prior is not None:
            overlaps = {}
            overlaps["Geo↔SPG"] = subspace_overlap(geometric_prior, semantic_prior)
            overlaps["Concat↔PF"] = subspace_overlap(concat_input, dense_prompt)
            for pair, o in overlaps.items():
                all_overlaps[pair].append(o)

        # Per-tile entry
        entry = {
            "tile_idx": tile_idx, "class_name": nodes["PF"]["class_name"],
            "iou_geo": iou_geo, "iou_spg": iou_spg, "iou_pf": iou_pf,
            "k90_geo": nodes["Geo"]["k_90"] if "Geo" in nodes else None,
            "k90_spg": nodes["SPG"]["k_90"],
            "k90_concat": nodes["Concat"]["k_90"] if "Concat" in nodes else None,
            "k90_pf": nodes["PF"]["k_90"],
        }
        per_tile.append(entry)

    # ── Aggregate across tiles ──
    summary: dict = {"n_tiles": len(per_tile)}

    # Aggregate node metrics
    node_agg = {}
    for node_name, node_list in all_nodes.items():
        agg = {}
        scalar_keys = [k for k, v in node_list[0].items()
                      if isinstance(v, (int, float)) and k not in ("tile_idx",)]
        for key in scalar_keys:
            vals = [d[key] for d in node_list if key in d]
            if vals:
                agg[key] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                    "min": round(float(np.min(vals)), 4),
                    "max": round(float(np.max(vals)), 4),
                }
        node_agg[node_name] = agg
    summary["nodes"] = node_agg

    # Aggregate CCA
    cca_agg = {}
    for pair_name, cca_list in all_ccas.items():
        cca_agg[pair_name] = {
            "mean_cca": round(float(np.mean([c["mean_cca"] for c in cca_list])), 4),
            "cos_global": round(float(np.mean([c["cos_global"] for c in cca_list])), 4),
            "mean_principal_angle_cos": round(float(np.mean([c["mean_principal_angle_cos"] for c in cca_list])), 4),
        }
    summary["cca"] = cca_agg

    # Aggregate subspace overlap
    overlap_agg = {}
    for pair_name, ov_list in all_overlaps.items():
        overlap_agg[pair_name] = {
            "grassmann_dist": round(float(np.mean([o["grassmann_dist_k8"] for o in ov_list])), 4),
            "subspace_overlap": round(float(np.mean([o["subspace_overlap_k8"] for o in ov_list])), 4),
        }
    summary["subspace_overlap"] = overlap_agg

    s_path, p_path = save_results(ctx.out_dir, summary, per_tile)

    # ── Print report ──
    print_header("Pipeline Information Flow — Complete Prompt Pathway")

    print("\n  ┌─ Effective Rank (K90) per Node{'─'*45}")
    header = f"  {'Node':<25s} {'K90':>6s}  {'K95':>6s}  {'K99':>6s}  {'Gini':>6s}  {'Ch Corr':>8s}  {'IoU':>8s}  {'SV1/SV2':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for node_name in ["Geo", "SPG", "Concat", "PF"]:
        if node_name in node_agg:
            a = node_agg[node_name]
            print(f"  {node_name:<25s} "
                  f"{a.get('k_90', {}).get('mean', 0):>6.1f}  "
                  f"{a.get('k_95', {}).get('mean', 0):>6.1f}  "
                  f"{a.get('k_99', {}).get('mean', 0):>6.1f}  "
                  f"{a.get('energy_gini', {}).get('mean', 0):>6.3f}  "
                  f"{a.get('ch_correlation', {}).get('mean', 0):>8.4f}  "
                  f"{a.get('iou', {}).get('mean', 0):>8.4f}  "
                  f"{a.get('sv_ratio_s1_s2', {}).get('mean', 0):>8.1f}")

    print("\n  ┌─ CCA Between Adjacent Nodes{'─'*49}")
    for pair_name, c in cca_agg.items():
        print(f"  {pair_name:<20s}  CCA={c['mean_cca']:.4f}  "
              f"Cos={c['cos_global']:.4f}  "
              f"PA_cos={c['mean_principal_angle_cos']:.4f}")

    if overlap_agg:
        print("\n  ┌─ Subspace Overlap (top-8 PCs){'─'*47}")
        for pair_name, o in overlap_agg.items():
            print(f"  {pair_name:<20s}  Grassmann={o['grassmann_dist']:.4f}  "
                  f"Overlap={o['subspace_overlap']:.4f}")

    # ── DIAGNOSIS ──
    print_header("DIAGNOSIS")

    concat_k90 = node_agg.get("Concat", {}).get("k_90", {}).get("mean", 0)
    pf_k90 = node_agg.get("PF", {}).get("k_90", {}).get("mean", 0)
    spg_k90 = node_agg.get("SPG", {}).get("k_90", {}).get("mean", 0)
    geo_k90 = node_agg.get("Geo", {}).get("k_90", {}).get("mean", 0)

    print(f"\n  Concat(Geo,SPG) K90 = {concat_k90:.1f}")
    print(f"  PF output       K90 = {pf_k90:.1f}")
    print(f"  ΔK90 (Concat → PF)   = {concat_k90 - pf_k90:+.1f}")

    if concat_k90 <= 3:
        print(f"\n  ★ Concat input IS already low-rank (K90={concat_k90:.1f})")
        print(f"    → The bottleneck is UPSTREAM of PromptFusion.")
        print(f"    → Geo (K90={geo_k90:.1f}) and SPG (K90={spg_k90:.1f}) are highly redundant.")
        print(f"    → Fix: improve SPG/GeoPrior to produce complementary representations.")
    elif pf_k90 <= 3 and concat_k90 > 5:
        print(f"\n  ★ Concat has sufficient rank ({concat_k90:.1f}), but PF collapses to {pf_k90:.1f}")
        print(f"    → PromptFusion IS the bottleneck.")
        print(f"    → Fix: redesign PromptFusion (residual, wider bottleneck, orthogonal constraint).")
    else:
        print(f"\n  ★ Both Concat ({concat_k90:.1f}) and PF ({pf_k90:.1f}) have moderate rank.")
        print(f"    → The issue may be more subtle than simple rank collapse.")
        print(f"    → Check CCA results for information loss patterns.")

    # Check the "PF learned wrong" hypothesis
    geo_iou = node_agg.get("Geo", {}).get("iou", {}).get("mean", 0)
    spg_iou = node_agg.get("SPG", {}).get("iou", {}).get("mean", 0)
    pf_iou = node_agg.get("PF", {}).get("iou", {}).get("mean", 0)
    print(f"\n  IoU: Geo={geo_iou:.4f}  SPG={spg_iou:.4f}  PF={pf_iou:.4f}")
    if pf_iou < min(geo_iou, spg_iou) - 0.001:
        print(f"  ⚠ PF IoU is WORSE than BOTH inputs → negative interference confirmed.")
        print(f"    PF is actively destroying information, not just failing to combine it.")

    print(f"\n[Summary] {s_path}")
    print(f"[Per-tile] {p_path}")
    print("[Done]")


if __name__ == "__main__":
    main()
