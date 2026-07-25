"""
Prompt 类别信息诊断 | Prompt Category Information Diagnostic.
==============================================================

三个 Hook 定位类别信息丢失的位置:
Three hooks to locate where category information is lost.

  Hook 1: Dense Prompt 有效秩 (channel variance / SVD / PCA)
  Hook 2: MaskDecoder Transformer 输入/输出特征判别力
  Hook 3: Sparse Token 类别判别力 (10×10 cosine similarity matrix)

用法 | Usage:
    # 不加载 Stage 2 ckpt — 只计算 raw cosine prompt (Hook 1 + 3)
    python tools/analysis/diag_prompt.py --fold 1 \
        --stage1-ckpt runs/stage1_fold1_seed42/best_adapter.pt

    # 加载 Stage 2 ckpt — 完整 Hook 1+2+3
    python tools/analysis/diag_prompt.py --fold 1 \
        --stage1-ckpt runs/stage1_fold1_seed42/best_adapter.pt \
        --stage2-ckpt runs/stage2_fold1_k5_seed42/best.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from adasam.adapters import CATAdapter
from adasam.backbone import build_mobile_sam, MobileSAMBackbone
from adasam.datasets.isaid_5i import ISAID5iDataset
from adasam.model.adasam_model import AdaSAMModel, AdaSAMModelConfig
from adasam.utils import set_seed
from adasam.utils.transforms import preprocess_image


# ═══════════════════════════════════════════════════════════════════
# Hook storage
# ═══════════════════════════════════════════════════════════════════

_hook_data: dict = {}


def _transformer_hook(module, args, output):
    """Capture transformer input/output.
    args: (src, pos_src, tokens)  — input
    output: (hs, src)              — output
    """
    _hook_data["transformer_src_in"] = args[0].detach().cpu()     # image+dense
    _hook_data["transformer_tokens_in"] = args[2].detach().cpu()  # sparse tokens
    _hook_data["transformer_hs_out"] = output[0].detach().cpu()   # updated tokens
    _hook_data["transformer_src_out"] = output[1].detach().cpu()  # updated dense


def _decoder_hook(module, args, output):
    """Capture mask_decoder forward output.
    output: (masks [B,1,H,W], iou_pred [B,1])
    """
    _hook_data["decoder_masks"] = output[0].detach().cpu()
    _hook_data["decoder_iou"] = output[1].detach().cpu()


# ═══════════════════════════════════════════════════════════════════
# Diagnostic functions
# ═══════════════════════════════════════════════════════════════════

def effective_rank(singular_values: np.ndarray, threshold: float = 0.99) -> int:
    """Number of singular values needed to explain `threshold` variance."""
    cumsum = np.cumsum(singular_values ** 2)
    total = cumsum[-1]
    if total < 1e-12:
        return 1
    return int(np.searchsorted(cumsum / total, threshold) + 1)


def analyze_dense_prompt(x: torch.Tensor) -> dict:
    """Hook 1: Analyze channel diversity of a [1, C, H, W] or [C, H, W] tensor."""
    if x.ndim == 4:
        x = x[0]
    C = x.shape[0]
    flat = x.reshape(C, -1).float()  # [C, N]

    ch_stds = flat.std(dim=1)        # [C]
    u, s, v = torch.svd(flat.float())
    s_np = s.cpu().numpy()
    total_var = (s_np ** 2).sum()

    return {
        "channel_std_mean": float(ch_stds.mean().item()),
        "channel_std_across": float(ch_stds.std().item()),
        "channel_std_min": float(ch_stds.min().item()),
        "channel_std_max": float(ch_stds.max().item()),
        "effective_rank_099": int(effective_rank(s_np)),
        "sv_top1_ratio": float((s_np[0] ** 2) / total_var if total_var > 0 else 1.0),
        "sv_top3_ratio": float((s_np[:3] ** 2).sum() / total_var if total_var > 0 else 1.0),
    }


def cosine_sim_matrix(tokens: dict[int, torch.Tensor]) -> dict:
    """Compute N×N cosine similarity matrix for per-class tokens."""
    classes = sorted(tokens.keys())
    n = len(classes)
    stacked = torch.cat([tokens[c].reshape(1, -1).float() for c in classes], dim=0)
    stacked_norm = F.normalize(stacked, dim=1)
    sim = (stacked_norm @ stacked_norm.T).cpu().numpy()

    diag = [sim[i, i] for i in range(n)]
    offdiag = [sim[i, j] for i in range(n) for j in range(n) if i != j]

    return {
        "matrix": sim,
        "classes": classes,
        "mean_diag": float(np.mean(diag)),
        "mean_offdiag": float(np.mean(offdiag)),
        "min_offdiag": float(np.min(offdiag)),
        "max_offdiag": float(np.max(offdiag)),
        "all_close_0.98": float(np.mean(offdiag)) > 0.98,
        "separation_ratio": float(np.mean(diag) / (np.mean(offdiag) + 1e-8)),
    }


# ═══════════════════════════════════════════════════════════════════
# Prompt computation (matches model internals)
# ═══════════════════════════════════════════════════════════════════

def compute_raw_cosine_prompt(
    query_features: torch.Tensor,
    support_features: torch.Tensor,
    support_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute raw cosine similarity prompt (matching raw_cosine_ablation path).

    Args:
        query_features: [1, C, H, W]
        support_features: [K, C, H, W]
        support_masks: [K, H, W]

    Returns:
        (dense_prompt [1, C, H, W], sparse_token [1, C], rsp_map [1, 1, H, W])
    """
    B, C, H, W = query_features.shape

    # Masked mean prototype (FG-only)
    masked = support_features * support_masks.unsqueeze(1)  # [K, C, H, W]
    fg_sum = masked.sum(dim=(0, 2, 3))                      # [C]
    fg_count = support_masks.sum() + 1e-8                   # scalar
    proto = fg_sum / fg_count                                # [C]

    # Cosine similarity
    q_flat = query_features.reshape(B, C, -1)                # [1, C, N]
    q_norm = F.normalize(q_flat, dim=1)                      # [1, C, N]
    p_norm = F.normalize(proto, dim=0)                       # [C]
    sim = torch.einsum("bcn,c->bn", q_norm, p_norm)          # [1, N]

    # Min-max normalize
    s_min = sim.min(dim=1, keepdim=True)[0]
    s_max = sim.max(dim=1, keepdim=True)[0]
    sim = (sim - s_min) / (s_max - s_min + 1e-8)
    rsp_map = sim.reshape(B, 1, H, W)                        # [1, 1, H, W]

    # Expand to C channels (ALL identical!)
    dense_prompt = rsp_map.expand(-1, C, -1, -1)             # [1, C, H, W]
    sparse_token = dense_prompt.mean(dim=(2, 3))             # [1, C]

    return dense_prompt, sparse_token, rsp_map


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Prompt category information diagnostic")
    parser.add_argument("--fold", type=int, default=1, help="fold index (0/1/2)")
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--stage1-ckpt", required=True,
                        help="path to Stage 1 adapter checkpoint")
    parser.add_argument("--stage2-ckpt", default=None,
                        help="path to Stage 2 checkpoint (enables Hook 2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default="data/iSAID-5i")
    parser.add_argument("--queries-per-class", type=int, default=3,
                        help="query tiles per class")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    # ── Data ──
    print(f"[data] loading iSAID-5i fold={args.fold} ...")
    dataset = ISAID5iDataset(
        root=args.data_root, fold=args.fold, split="val",
    )
    val_classes = dataset.visible_classes()
    print(f"[data] val classes: {val_classes}")

    # ── Backbone + Adapter ──
    print("[model] building backbone + adapter ...")
    sam = build_mobile_sam(checkpoint="weights/mobile_sam.pt", device=device)
    backbone = MobileSAMBackbone(sam.image_encoder).to(device).eval()

    adapter = CATAdapter(dim=256).to(device)
    ckpt = torch.load(args.stage1_ckpt, map_location=device, weights_only=True)
    adapter.load_state_dict(ckpt["adapter"])
    adapter.eval()
    print(f"  stage1 epoch={ckpt.get('epoch', '?')} fold={ckpt.get('fold', '?')}")

    # ── Stage 2 model (optional) ──
    model: AdaSAMModel | None = None
    transformer_hook = None
    decoder_hook = None

    if args.stage2_ckpt is not None:
        print(f"[load_stage2] loading from: {args.stage2_ckpt}")
        st2_ckpt = torch.load(args.stage2_ckpt, map_location=device, weights_only=False)
        cfg = AdaSAMModelConfig.from_dict(st2_ckpt["config"])
        # Need to rebuild Sam for the model (it uses prompt_encoder + mask_decoder)
        sam_full = build_mobile_sam(checkpoint="weights/mobile_sam.pt", device=device)
        model = AdaSAMModel(sam_full, cfg).to(device)
        model.load_state_dict(st2_ckpt["model"])
        model.eval()

        # Register hooks on MaskDecoder's transformer
        md = model.sam_decoder.mask_decoder
        transformer_hook = md.transformer.register_forward_hook(_transformer_hook)
        decoder_hook = md.register_forward_hook(_decoder_hook)
        print(f"  cfg.raw_cosine_ablation = {cfg.raw_cosine_ablation}")
        print(f"  cfg.use_geometric_prior = {cfg.use_geometric_prior}")
        print("[hooks] registered on mask_decoder.transformer")
    else:
        print("[model] no Stage 2 checkpoint — raw cosine prompt only (Hook 1+3)")

    # ═══════════════════════════════════════════════════════════════
    # Per-class support cache
    # ═══════════════════════════════════════════════════════════════
    print("[support] building per-class support cache ...")

    support_cache: dict[int, dict] = {}
    for cls_id in tqdm(val_classes, desc="support cache"):
        tiles = dataset.class_to_tiles(cls_id)
        if len(tiles) < args.k_shot + 1:
            print(f"  [warn] class {cls_id}: only {len(tiles)} tiles")
            support_cache[cls_id] = None
            continue

        # Fixed support tiles for this diagnostic run
        sampled = rng.sample(tiles, args.k_shot)
        s_feats, s_masks = [], []
        for idx in sampled:
            sample = dataset[idx]
            x, _ = preprocess_image(sample["image"])
            x = x.unsqueeze(0).to(device)
            with torch.no_grad():
                emb = backbone(x)["image_embedding"]
                emb = adapter(emb)
            mask = dataset.get_class_mask(idx, cls_id).to(device)
            mask_64 = F.interpolate(
                mask[None, None].float(), size=(64, 64), mode="nearest"
            )[0, 0]
            s_feats.append(emb[0])
            s_masks.append(mask_64)

        support_cache[cls_id] = {
            "feats": torch.stack(s_feats, dim=0),   # [K, 256, 64, 64]
            "masks": torch.stack(s_masks, dim=0),   # [K, 64, 64]
            "support_tile_indices": sampled,
        }

    # ═══════════════════════════════════════════════════════════════
    # Per-class diagnostic
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("Per-class diagnostics ...")
    print("=" * 72)

    # Accumulators
    dense_rank_all: dict[int, list] = {c: [] for c in val_classes}
    sparse_tokens_all: dict[int, list] = {c: [] for c in val_classes}
    sparse_after_trans_all: dict[int, list] = {c: [] for c in val_classes}
    dense_after_trans_all: dict[int, list] = {c: [] for c in val_classes}

    for cls_id in tqdm(val_classes, desc="per-class"):
        sup = support_cache.get(cls_id)
        if sup is None:
            continue

        # Query tiles: from the SAME class, not used as support
        tiles = dataset.class_to_tiles(cls_id)
        query_pool = [t for t in tiles if t not in sup["support_tile_indices"]]
        if len(query_pool) < 1:
            print(f"  [warn] class {cls_id}: no query tiles after excluding support")
            continue

        n_queries = min(args.queries_per_class, len(query_pool))
        query_indices = rng.sample(query_pool, n_queries)

        for q_idx in query_indices:
            sample = dataset[q_idx]
            x, _ = preprocess_image(sample["image"])
            x = x.unsqueeze(0).to(device)
            with torch.no_grad():
                q_emb = backbone(x)["image_embedding"]
                q_emb = adapter(q_emb)

            s_feats = sup["feats"].to(device)
            s_masks = sup["masks"].to(device)

            if model is not None:
                # Run through actual model → hooks fire automatically
                _hook_data.clear()
                with torch.no_grad():
                    spg_out, low_res, iou_pred = model.forward_train(
                        q_emb, s_feats, s_masks
                    )

                # Extract dense_prompt from the model's forward_train path
                # For raw_cosine_ablation: computed inline in forward_train
                # For full model: need to capture from intermediate
                if model.cfg.raw_cosine_ablation:
                    # Re-compute to get exact values (cheap, no grad)
                    dense_prompt, sparse_token, rsp_map = compute_raw_cosine_prompt(
                        q_emb, s_feats, s_masks
                    )
                else:
                    # Use SPG output as approximation
                    dense_prompt = spg_out.semantic_prior
                    sparse_token = dense_prompt.mean(dim=(2, 3))
            else:
                dense_prompt, sparse_token, rsp_map = compute_raw_cosine_prompt(
                    q_emb, s_feats, s_masks
                )

            # Hook 1: dense prompt rank
            dense_rank_all[cls_id].append(analyze_dense_prompt(dense_prompt))

            # Hook 3: sparse token
            sparse_tokens_all[cls_id].append(sparse_token.cpu())

            # Hook 2: transformer internals
            if "transformer_hs_out" in _hook_data:
                # hs_out: [1, N_tokens, C]
                # tokens = [iou(1), mask(4), sparse(1)] = 6 tokens total
                # sparse is the last token (index 5 for multimask=False → 1+4+1=6)
                hs = _hook_data["transformer_hs_out"]  # [1, N, C]
                sparse_after_trans_all[cls_id].append(hs[0, -1, :])  # last token

            if "transformer_src_out" in _hook_data:
                dense_after_trans_all[cls_id].append(
                    _hook_data["transformer_src_out"]  # [1, C, H, W]
                )

    # ═══════════════════════════════════════════════════════════════
    # Aggregation
    # ═══════════════════════════════════════════════════════════════

    def avg_stats(per_class_list: dict[int, list]) -> dict[int, dict]:
        result = {}
        for cls_id, items in per_class_list.items():
            if not items:
                continue
            avg = {}
            for k in items[0]:
                vals = [it[k] for it in items]
                avg[k] = float(np.mean(vals)) if isinstance(vals[0], (int, float, np.floating)) else vals[0]
            result[cls_id] = avg
        return result

    def mean_tensor(per_class_list: dict[int, list]) -> dict[int, torch.Tensor]:
        result = {}
        for cls_id, items in per_class_list.items():
            if items:
                result[cls_id] = torch.stack([t.reshape(-1) for t in items]).mean(dim=0)
        return result

    dense_rank_avg = avg_stats(dense_rank_all)
    sparse_avg = mean_tensor(sparse_tokens_all)
    sparse_trans_avg = mean_tensor(sparse_after_trans_all)

    # ═══════════════════════════════════════════════════════════════
    # Reports
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 72)
    print("HOOK 1: Dense Prompt Effective Rank (before MaskDecoder)")
    print("=" * 72)
    print(f"  {'class':>6}  {'rank_099':>8}  {'top1_sv':>8}  {'top3_sv':>8}  "
          f"{'ch_std_mean':>12}  {'ch_std_across':>14}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*14}")
    for cls_id in sorted(dense_rank_avg.keys()):
        r = dense_rank_avg[cls_id]
        print(f"  {cls_id:>6}  {r['effective_rank_099']:>8.0f}  {r['sv_top1_ratio']:>8.4f}  "
              f"{r['sv_top3_ratio']:>8.4f}  {r['channel_std_mean']:>12.6f}  "
              f"{r['channel_std_across']:>14.6f}")

    all_top1 = [r["sv_top1_ratio"] for r in dense_rank_avg.values()]
    all_ranks = [r["effective_rank_099"] for r in dense_rank_avg.values()]
    all_ch_std = [r["channel_std_across"] for r in dense_rank_avg.values()]
    print(f"\n  Summary: mean rank={np.mean(all_ranks):.1f}, "
          f"mean top1_sv={np.mean(all_top1):.4f}, "
          f"mean ch_std_across={np.mean(all_ch_std):.6f}")

    if np.mean(all_top1) > 0.99 and np.mean(all_ch_std) < 1e-4:
        print("  ⚠ DENSE PROMPT RANK≈1: all 256 channels are near-identical!")
        print("    → Dense prompt is just a 1-channel similarity map broadcast 256 times.")

    print("\n" + "=" * 72)
    print("HOOK 3: Sparse Token Category Discriminability")
    print("=" * 72)
    if len(sparse_avg) >= 2:
        r = cosine_sim_matrix(sparse_avg)
        print(f"  classes: {r['classes']}")
        print(f"  separation_ratio = {r['separation_ratio']:.4f}  (>1.0 = discriminative)")
        print(f"  mean_diag    = {r['mean_diag']:.4f}")
        print(f"  mean_offdiag = {r['mean_offdiag']:.4f}  [{r['min_offdiag']:.4f}, {r['max_offdiag']:.4f}]")
        print(f"  all_close(0.98) = {r['all_close_0.98']}")

        if r["all_close_0.98"]:
            print("  ⚠ SPARSE TOKEN COLLAPSE — all classes produce near-identical sparse tokens!")
            print("    → Sparse token carries ZERO category identity.")
            print("    → Decoder has no way to know WHICH class to segment.")
        elif r["separation_ratio"] > 1.5:
            print("  ✓ Sparse tokens encode category identity well.")
        else:
            print("  ~ Sparse tokens are weakly discriminative.")

        # Print matrix
        print("\n  Cosine similarity matrix:")
        header = "      " + "  ".join(f"c{str(c):>2s}" for c in [str(cc) for cc in r["classes"]])
        # pad short class names
        header = "       " + "  ".join(f"{c:>3}" for c in r["classes"])
        print(f"  {header}")
        for i, ci in enumerate(r["classes"]):
            row = "  ".join(f"{r['matrix'][i, j]:.3f}" for j in range(len(r["classes"])))
            print(f"  {ci:>3}  {row}")

    if sparse_trans_avg:
        print("\n" + "=" * 72)
        print("HOOK 2b: Sparse Token AFTER Transformer (updated by cross-attention)")
        print("=" * 72)
        if len(sparse_trans_avg) >= 2:
            r2 = cosine_sim_matrix(sparse_trans_avg)
            print(f"  separation_ratio = {r2['separation_ratio']:.4f}")
            print(f"  mean_offdiag = {r2['mean_offdiag']:.4f}")
            print(f"  all_close(0.98) = {r2['all_close_0.98']}")
            if r2["all_close_0.98"]:
                print("  ⚠ TRANSFORMER DID NOT HELP — sparse tokens still collapsed after cross-attention!")
            elif r2["separation_ratio"] > 1.5:
                print("  ✓ Transformer cross-attention created category-discriminative sparse tokens.")
            else:
                print("  ~ Weak improvement after transformer.")
        else:
            print("  [skip] not enough classes")

    if dense_after_trans_all:
        print("\n" + "=" * 72)
        print("HOOK 2a: Dense Features AFTER Transformer")
        print("=" * 72)
        dense_trans_rank = {}
        for cls_id, items in dense_after_trans_all.items():
            if items:
                ranks = [analyze_dense_prompt(t) for t in items]
                dense_trans_rank[cls_id] = {
                    k: float(np.mean([r[k] for r in ranks]))
                    for k in ranks[0] if isinstance(ranks[0][k], (int, float, np.floating))
                }
        for cls_id in sorted(dense_trans_rank.keys()):
            r = dense_trans_rank[cls_id]
            print(f"  class {cls_id:2d}: rank_099={r['effective_rank_099']:4.0f}  "
                  f"top1_sv={r['sv_top1_ratio']:.4f}  ch_std={r['channel_std_mean']:.4f}")

        # Cross-class GAP cosine
        gap_feats = {}
        for cls_id, items in dense_after_trans_all.items():
            if items:
                # mean pool over queries then over spatial
                stacked = torch.cat([t.reshape(1, -1) for t in items], dim=0)
                gap_feats[cls_id] = stacked.mean(dim=0)
        if len(gap_feats) >= 2:
            r3 = cosine_sim_matrix(gap_feats)
            print(f"\n  Dense GAP cross-class cosine after transformer:")
            print(f"  separation_ratio = {r3['separation_ratio']:.4f}")
            print(f"  mean_offdiag = {r3['mean_offdiag']:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 72)

    rank_ok = np.mean(all_top1) < 0.99 or np.mean(all_ch_std) > 1e-3
    sparse_ok = len(sparse_avg) >= 2 and not cosine_sim_matrix(sparse_avg)["all_close_0.98"]

    print(f"""
  Hook 1 — Dense Prompt effective rank:   {'✓ OK' if rank_ok else '✗ RANK≈1 — channels identical'}
  Hook 3 — Sparse Token discriminability: {'✓ OK' if sparse_ok else '✗ COLLAPSED — all classes identical'}

  Interpretation:
  {'→ Both are broken: category info never reaches MaskDecoder.' if not rank_ok and not sparse_ok else ''}
  {'→ Dense prompt is degenerate but sparse might carry signal.' if not rank_ok and sparse_ok else ''}
  {'→ Dense prompt has channel diversity — need to check Hook 2 for where info is lost.' if rank_ok and not sparse_ok else ''}
  {'→ Prompt structure looks healthy — bottleneck might be in loss/supervision.' if rank_ok and sparse_ok else ''}
""")

    # Cleanup
    if transformer_hook:
        transformer_hook.remove()
    if decoder_hook:
        decoder_hook.remove()


if __name__ == "__main__":
    main()
