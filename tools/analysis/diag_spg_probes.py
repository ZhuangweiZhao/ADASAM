"""
SPG Probe 诊断 | SPG Probe Diagnostic.
========================================

核心问题：SPG 的 16 个 probe 分别在编码什么？是语义（class-specific）还是
空间模式（objectness/location/boundary）？

方法：手动复现 SPG forward，捕获每层 per-probe 内部状态，然后做三项分析：
  ① Probe Spatial Diversity — 每个 probe 的 mask 空间模式、probe 间 IoU
  ② Probe Feature Orthogonality — probe embedding / 条件化后特征的 Gram 矩阵
  ③ Probe-Class Sensitivity — per-probe 激活与 support class 的 ANOVA F-score

如果 16 个 probe 全部编码相似的 object-boundary 模式、probe-probe cosine≈1、
class ANOVA F≈0，则确认 "SPG = Objectness Generator" 假设。

用法 | Usage:
    python tools/analysis/diag_spg_probes.py \
        --stage2-ckpt runs/stage2_fold1_k5_seed42/best_model.pt \
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


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def build_support_for_class(
    dataset, class_id: int, k_shot: int, device: torch.device,
    backbone: MobileSAMBackbone, adapter, rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor] | None:
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
        [resize_mask(m, (feats.shape[2], feats.shape[3])).to(device) for m in masks], dim=0)
    if masks_grid.sum() < 1.0:
        return None
    return feats, masks_grid


def run_spg_step_by_step(spg, query_features, support_memory, dense_pe):
    """Manually execute SPG forward, capturing per-probe internal states at each layer.

    Returns:
        per_layer: list of dicts, one per layer, each with:
            - probe_logits: [N] probe confidences
            - masks: [N, gh, gw] per-probe mask logits
            - q: [1, N, C] probe features
            - attn_mask_stats: dict with 'frac_masked' (fraction of pixels masked out)
    """
    gh, gw = query_features.shape[2], query_features.shape[3]
    N = spg.cfg.num_probes
    L = spg.cfg.num_layers

    mask_features = query_features[0].clone()  # [C, gh, gw]
    memory = query_features.flatten(2).permute(0, 2, 1)  # [1, gh*gw, C]
    memory_pe = dense_pe.flatten(2).permute(0, 2, 1)     # [1, gh*gw, C]
    probe_pos = spg.probe_pos.weight.unsqueeze(0)         # [1, N, C]

    q = spg.probe_feat.weight.unsqueeze(0)                # [1, N, C]

    has_support = support_memory.shape[0] > 0
    support_key = support_memory.unsqueeze(0) if has_support else None  # [1, M, C]

    per_layer = []

    # Initial prediction (before any layer processing)
    probe_logits, masks = spg._predict(q[0], mask_features)

    for i in range(L):
        # ── Save state BEFORE layer processing (after _predict from previous iter) ──
        attn_mask = spg._build_attn_mask(masks, spg.cfg.num_heads)
        # attn_mask shape: [num_heads, N, gh*gw], True = MASKED OUT
        frac_masked = float(attn_mask[0].float().mean().item())

        per_layer.append({
            "probe_logits": probe_logits.detach().cpu().clone(),  # [N]
            "masks": masks.detach().cpu().clone(),                 # [N, gh, gw]
            "q": q.detach().cpu().clone(),                         # [1, N, C]
            "attn_frac_masked": frac_masked,
        })

        # 1. Masked cross-attention (probes → image features)
        out, _ = spg.cross_attn[i](
            query=q + probe_pos, key=memory + memory_pe, value=memory,
            attn_mask=attn_mask, need_weights=False,
        )
        q = spg.cross_norm[i](q + out)

        # 2. Support cross-attention (probes → support memory)
        if has_support:
            out_s, _ = spg.cross_attn_support[i](
                query=q + probe_pos, key=support_key, value=support_key,
                need_weights=False,
            )
            q = spg.support_cross_norm[i](q + out_s)

        # 3. Self-attention (probe ↔ probe)
        out, _ = spg.self_attn[i](
            query=q + probe_pos, key=q + probe_pos, value=q, need_weights=False,
        )
        q = spg.self_norm[i](q + out)

        # 4. FFN
        q = spg.ffn_norm[i](q + spg.ffn[i](q))

        # Re-predict
        probe_logits, masks = spg._predict(q[0], mask_features)

        # Inter-layer mask feedback
        if spg.feedback_conv is not None and i < L - 1:
            masks_prob = masks.detach().sigmoid()
            fg_conf = masks_prob.max(dim=0)[0]
            probe_w = probe_logits.detach().sigmoid()
            weighted = (masks_prob * probe_w[:, None, None]).sum(dim=0)
            extra = torch.stack([fg_conf, weighted], dim=0).unsqueeze(0)
            feedback = spg.feedback_conv[i](
                torch.cat([mask_features.unsqueeze(0), extra], dim=1)
            )[0]
            mask_features = mask_features + feedback

    # Save final state (after last layer)
    per_layer.append({
        "probe_logits": probe_logits.detach().cpu().clone(),
        "masks": masks.detach().cpu().clone(),
        "q": q.detach().cpu().clone(),
        "attn_frac_masked": -1.0,  # no attention after last layer
    })

    return per_layer  # L+1 entries (initial + after each layer)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SPG Probe Diagnostic")
    parser.add_argument("--stage2-ckpt", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--n-queries", type=int, default=30,
                        help="Number of multi-class query tiles")
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
    for p in model.parameters():
        p.requires_grad_(False)

    adapter_state = ckpt.get("cat_adapter")
    if adapter_state is not None:
        adapter_cfg = ckpt.get("config", {}).get("adapter", {})
        adapter = CATAdapter(
            dim=256, bottleneck=int(adapter_cfg.get("bottleneck", 64)),
        ).to(device)
        adapter.load_state_dict(adapter_state)
        adapter.eval()
        for p in adapter.parameters():
            p.requires_grad_(False)
    else:
        adapter = None

    spg = model.spg
    if spg is None:
        print("ERROR: SPG not found in model")
        sys.exit(1)
    N_PROBES = spg.cfg.num_probes
    N_LAYERS = spg.cfg.num_layers
    print(f"SPG: {N_PROBES} probes, {N_LAYERS} layers")

    data_root = str(_REPO_ROOT / args.data_root) if not Path(args.data_root).is_absolute() \
        else args.data_root
    dataset = ISAID5iDataset(
        root=data_root, fold=args.fold, split="val", mode="base")
    val_classes = sorted(dataset.visible_classes())
    cls_to_idx = {cls: i for i, cls in enumerate(val_classes)}
    idx_to_name = {i: ISAID5I_CATEGORIES.get(cls, str(cls))
                   for cls, i in cls_to_idx.items()}

    # ── Find multi-class query tiles ──
    tile_scores = []
    for idx in range(len(dataset)):
        classes_present = []
        for cls in val_classes:
            gt = dataset.get_class_mask(idx, cls)
            if gt is not None and gt.sum() > 100:
                classes_present.append(cls)
        if len(classes_present) >= 2:
            tile_scores.append((idx, len(classes_present), classes_present))
    tile_scores.sort(key=lambda x: -x[1])
    n_queries = min(args.n_queries, len(tile_scores))
    selected = tile_scores[:n_queries]
    print(f"Multi-class tiles: {len(tile_scores)}, using {n_queries}")

    # ═══════════════════════════════════════════════════════════════════
    # Collect per-probe states
    # ═══════════════════════════════════════════════════════════════════

    # Per (tile, class) sample: list of (layer_idx, probe_idx) data
    # all_samples[sample_idx] = {
    #     "class": int,
    #     "class_name": str,
    #     "tile_idx": int,
    #     "per_layer": per_layer,  # from run_spg_step_by_step
    # }
    all_samples: list[dict] = []

    print(f"\n{'=' * 72}")
    print(f"Collecting per-probe SPG states ({N_LAYERS} layers × {N_PROBES} probes)")
    print("=" * 72)

    for tile_idx, n_cls, cls_list in tqdm(selected, desc="collecting"):
        sample = dataset[tile_idx]
        x, _ = preprocess_image(sample["image"])

        with torch.no_grad():
            q_emb = backbone(x.unsqueeze(0).to(device))["image_embedding"]
            if adapter is not None:
                q_emb = adapter(q_emb)
            dense_pe = model.sam_decoder.prompt_encoder.get_dense_pe()

            for sup_cls in cls_list[:4]:
                sup_data = build_support_for_class(
                    dataset, sup_cls, args.k_shot, device, backbone, adapter, rng)
                if sup_data is None:
                    continue
                sup_feat, sup_mask = sup_data
                support_memory = model.support_encoder(sup_feat, sup_mask)

                # Manual SPG step-by-step to capture probe internals
                per_layer = run_spg_step_by_step(
                    spg, q_emb, support_memory, dense_pe)

                all_samples.append({
                    "class": sup_cls,
                    "class_name": ISAID5I_CATEGORIES.get(sup_cls, str(sup_cls)),
                    "tile_idx": tile_idx,
                    "per_layer": per_layer,
                })

                del sup_feat, sup_mask, support_memory

        del q_emb, dense_pe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    N_SAMPLES = len(all_samples)
    print(f"\nCollected {N_SAMPLES} samples ({N_SAMPLES} query × class pairs)")
    if N_SAMPLES < 10:
        print("ERROR: too few samples")
        sys.exit(1)

    # Count per class
    class_counts = defaultdict(int)
    for s in all_samples:
        class_counts[s["class"]] += 1
    print(f"Per class: { {idx_to_name.get(c, str(c)): n for c, n in class_counts.items()} }")

    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS 1: Probe Spatial Diversity
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'=' * 72}")
    print("ANALYSIS 1: Probe Spatial Diversity")
    print("  Q: Do probes learn distinct spatial patterns, or are they all the same?")
    print("=" * 72)

    # Use final layer (last per_layer entry, index N_LAYERS) for main analysis
    FINAL = N_LAYERS  # index of final state in per_layer

    # 1a. Per-probe average mask (across all samples)
    all_masks = torch.stack(
        [s["per_layer"][FINAL]["masks"] for s in all_samples], dim=0)  # [S, N, gh, gw]
    probe_mean_mask = all_masks.sigmoid().mean(dim=0)  # [N, gh, gw]

    # 1b. Per-probe average coverage (% of spatial area above threshold)
    threshold = 0.5
    probe_coverage = (all_masks.sigmoid() > threshold).float().mean(dim=(0, 2, 3))  # [N]

    print(f"\n  Per-probe coverage (fraction of pixels with sigmoid>0.5, avg over {N_SAMPLES} samples):")
    print(f"  {'Probe':<8} {'Coverage':>10} {'Bar'}")
    print(f"  {'-' * 35}")
    for p in range(N_PROBES):
        bar = "█" * max(1, int(probe_coverage[p] * 60))
        print(f"  {p:<8} {probe_coverage[p]:>10.4f}  {bar}")
    print(f"  {'Mean':<8} {probe_coverage.mean():>10.4f}")
    print(f"  {'Std':<8} {probe_coverage.std():>10.4f}")

    # 1c. Pairwise IoU between probe masks
    pairwise_iou = torch.zeros(N_PROBES, N_PROBES)
    for i in range(N_PROBES):
        for j in range(N_PROBES):
            mask_i = all_masks[:, i].sigmoid() > threshold  # [S, gh, gw]
            mask_j = all_masks[:, j].sigmoid() > threshold
            inter = (mask_i & mask_j).float().sum(dim=(1, 2))
            union = (mask_i | mask_j).float().sum(dim=(1, 2))
            iou_per_sample = inter / union.clamp(min=1)
            pairwise_iou[i, j] = iou_per_sample.mean()

    # Average off-diagonal IoU
    off_diag_mask = ~torch.eye(N_PROBES, dtype=bool)
    mean_pairwise_iou = pairwise_iou[off_diag_mask].mean().item()
    mean_self_iou = pairwise_iou.diag().mean().item()

    print(f"\n  Probe-Probe IoU Matrix (avg over {N_SAMPLES} samples):")
    print(f"  Mean self-IoU (diagonal):  {mean_self_iou:.4f}")
    print(f"  Mean cross-IoU (off-diag): {mean_pairwise_iou:.4f}")
    print(f"  Cross / Self ratio:        {mean_pairwise_iou / max(mean_self_iou, 1e-8):.4f}")

    if mean_pairwise_iou > 0.7:
        print(f"  ⚠️  DIAGNOSIS: Probes are highly overlapping → SPATIAL COLLAPSE")
    elif mean_pairwise_iou > 0.4:
        print(f"  ⚠️  DIAGNOSIS: Moderate overlap — probes share significant regions")
    else:
        print(f"  ✅ DIAGNOSIS: Probes have distinct spatial patterns")

    # Print IoU matrix as compact table
    print(f"\n  IoU Matrix (final layer):")
    print(f"  {'':>6}", end="")
    for j in range(min(N_PROBES, 16)):
        print(f"{j:>6}", end="")
    print()
    for i in range(N_PROBES):
        print(f"  {i:>5} ", end="")
        for j in range(N_PROBES):
            v = pairwise_iou[i, j].item()
            if i == j:
                print(f"  *   ", end="")
            else:
                print(f"{v:>5.2f} ", end="")
        print()

    # 1d. Per-probe spatial center of mass
    gh, gw = all_masks.shape[2], all_masks.shape[3]
    y_grid, x_grid = torch.meshgrid(
        torch.arange(gh, dtype=torch.float32),
        torch.arange(gw, dtype=torch.float32),
        indexing="ij",
    )
    probe_centers = []
    for p in range(N_PROBES):
        weights = probe_mean_mask[p]  # [gh, gw]
        total = weights.sum()
        if total > 1e-8:
            cy = (weights * y_grid).sum() / total
            cx = (weights * x_grid).sum() / total
        else:
            cy, cx = gh / 2, gw / 2
        probe_centers.append((float(cy), float(cx)))

    print(f"\n  Probe spatial centers (y, x) in [{gh}×{gw}] grid:")
    for p in range(N_PROBES):
        cy, cx = probe_centers[p]
        # quadrant
        qy = "top" if cy < gh / 2 else "bottom"
        qx = "left" if cx < gw / 2 else "right"
        print(f"    Probe {p:>2}: ({cy:>5.1f}, {cx:>5.1f}) → {qy}-{qx}")

    # Spread of centers
    centers_arr = np.array(probe_centers)
    center_spread = np.std(centers_arr, axis=0)
    print(f"  Center spread (std): y={center_spread[0]:.1f}, x={center_spread[1]:.1f}")
    if center_spread.mean() < 3:
        print(f"  ⚠️  DIAGNOSIS: All probes center on same region — no spatial diversity")
    else:
        print(f"  ✅ DIAGNOSIS: Probes have diverse spatial centers")

    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS 2: Probe Feature Orthogonality
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'=' * 72}")
    print("ANALYSIS 2: Probe Feature Orthogonality")
    print("  Q: Are probe features (after conditioning) collapsed or diverse?")
    print("=" * 72)

    # 2a. Cosine similarity of learned probe_feat embeddings (pre-conditioning)
    probe_feat = spg.probe_feat.weight.data.cpu()  # [N, C]
    probe_feat_n = F.normalize(probe_feat, dim=1)
    feat_cos = probe_feat_n @ probe_feat_n.T  # [N, N]

    off_diag_feat = feat_cos[off_diag_mask].mean().item()
    print(f"\n  2a. probe_feat (learned embedding, pre-conditioning):")
    print(f"      Mean cos(probe_i, probe_j), i≠j: {off_diag_feat:.6f}")
    if off_diag_feat > 0.95:
        print(f"      ⚠️  DIAGNOSIS: probe_feat embeddings are collapsed (cos≈1)")
    elif off_diag_feat > 0.7:
        print(f"      ⚠️  DIAGNOSIS: Weak diversity in probe_feat")
    else:
        print(f"      ✅ DIAGNOSIS: probe_feat has feature diversity")

    # 2b. Cosine similarity of conditioned probe features q (after SPG layers)
    all_q = torch.stack(
        [s["per_layer"][FINAL]["q"][0] for s in all_samples], dim=0)  # [S, N, C]
    q_mean = all_q.mean(dim=0)  # [N, C] — average probe feature across all samples
    q_mean_n = F.normalize(q_mean, dim=1)
    q_cos = q_mean_n @ q_mean_n.T  # [N, N]

    off_diag_q = q_cos[off_diag_mask].mean().item()
    print(f"\n  2b. q (conditioned probe features, after all layers, avg over {N_SAMPLES} samples):")
    print(f"      Mean cos(q_i, q_j), i≠j: {off_diag_q:.6f}")
    if off_diag_q > 0.95:
        print(f"      ⚠️  DIAGNOSIS: conditioned probes collapse (cos≈1) → all probes encode same thing")
    elif off_diag_q > 0.7:
        print(f"      ⚠️  DIAGNOSIS: Weak feature diversity in conditioned probes")
    else:
        print(f"      ✅ DIAGNOSIS: Conditioned probes maintain feature diversity")

    # 2c. Per-layer evolution of probe feature cosine
    print(f"\n  2c. Per-layer evolution of mean off-diag cosine:")
    print(f"      {'Layer':<8} {'cos(probe_feat)':>18} {'cos(q_cond)':>18}")
    print(f"      {'-' * 48}")
    # Layer -1 = pre-conditioning (just probe_feat)
    print(f"      {'init':<8} {off_diag_feat:>18.6f} {'—':>18}")
    for layer_idx in range(N_LAYERS + 1):
        all_q_l = torch.stack(
            [s["per_layer"][layer_idx]["q"][0] for s in all_samples], dim=0)  # [S, N, C]
        q_l_mean = all_q_l.mean(dim=0)
        q_l_n = F.normalize(q_l_mean, dim=1)
        q_l_cos = (q_l_n @ q_l_n.T)[off_diag_mask].mean().item()
        label = f"layer {layer_idx}" if layer_idx < N_LAYERS else f"final"
        print(f"      {label:<8} {'—':>18} {q_l_cos:>18.6f}")

    # 2d. Gram matrix visualization summary
    print(f"\n  2d. probe_feat Gram matrix (16×16, diagonal=1.0):")
    for i in range(N_PROBES):
        row = " ".join(f"{feat_cos[i, j]:.3f}" for j in range(N_PROBES))
        print(f"      {row}")

    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS 3: Probe-Class Sensitivity (MOST IMPORTANT)
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'=' * 72}")
    print("ANALYSIS 3: Probe-Class Sensitivity (ANOVA F-score)")
    print("  Q: Does probe activation CHANGE when support class changes?")
    print("  If F≈1 for all probes → probes encode objectness, not semantics")
    print("=" * 72)

    # For each (sample, probe), compute a scalar activation = mean mask value
    # Then compute ANOVA: between-class variance / within-class variance
    activations = torch.zeros(N_SAMPLES, N_PROBES)  # [S, N]
    for si, s in enumerate(all_samples):
        masks = s["per_layer"][FINAL]["masks"]  # [N, gh, gw]
        activations[si] = masks.sigmoid().mean(dim=(1, 2))  # [N]

    class_labels = np.array([s["class"] for s in all_samples])
    unique_classes = sorted(set(class_labels))

    print(f"\n  3a. Per-probe ANOVA F-score (activation ~ class):")
    print(f"      {'Probe':<8} {'F-score':>10} {'p-value':>10} {'Sensitive?':>15} {'Best class':>15}")
    print(f"      {'-' * 65}")

    probe_f_scores = []
    for p in range(N_PROBES):
        act_p = activations[:, p].numpy()  # [S]

        # Compute group means
        grand_mean = act_p.mean()
        ss_between = 0.0
        ss_within = 0.0
        group_means = {}
        for cls in unique_classes:
            mask_cls = class_labels == cls
            group_act = act_p[mask_cls]
            group_mean = group_act.mean()
            group_means[cls] = group_mean
            n_k = len(group_act)
            ss_between += n_k * (group_mean - grand_mean) ** 2
            ss_within += ((group_act - group_mean) ** 2).sum()

        df_between = len(unique_classes) - 1
        df_within = N_SAMPLES - len(unique_classes)

        ms_between = ss_between / max(df_between, 1)
        ms_within = ss_within / max(df_within, 1)
        f_score = ms_between / max(ms_within, 1e-8)

        # Approximate p-value from F-distribution (using scipy if available, else rough)
        # For quick diagnosis: F > 2 is roughly significant for df~10,100
        if f_score > 5:
            diag = "✅ CLASS-SENSITIVE"
        elif f_score > 2:
            diag = "⚠️  WEAKLY sensitive"
        else:
            diag = "❌ NOT sensitive"

        best_cls = max(group_means, key=group_means.get)
        best_name = ISAID5I_CATEGORIES.get(best_cls, str(best_cls))

        print(f"      {p:<8} {f_score:>10.2f} {'—':>10} {diag:<15} {best_name:<15}")
        probe_f_scores.append(f_score)

    mean_f = np.mean(probe_f_scores)
    max_f = np.max(probe_f_scores)
    n_sensitive = sum(1 for f in probe_f_scores if f > 5)

    print(f"\n  Summary:")
    print(f"    Mean F-score: {mean_f:.2f}")
    print(f"    Max  F-score: {max_f:.2f}")
    print(f"    Probes with F>5 (class-sensitive): {n_sensitive}/{N_PROBES}")

    if n_sensitive == 0:
        print(f"\n  ‼️  DIAGNOSIS: ZERO probes are class-sensitive!")
        print(f"      All {N_PROBES} probes encode objectness/location, not semantics.")
        print(f"      SPG = Objectness Generator (confirmed)")
    elif n_sensitive <= 3:
        print(f"\n  ⚠️  DIAGNOSIS: Only {n_sensitive}/{N_PROBES} probes carry class info")
        print(f"      Vast majority encode objectness")
    else:
        print(f"\n  ✅ DIAGNOSIS: {n_sensitive}/{N_PROBES} probes are class-sensitive")

    # 3b. Per-class probe ranking (which probes fire most for each class)
    print(f"\n  3b. Per-class top-3 probes (by mean activation):")
    for cls in unique_classes:
        mask_cls = class_labels == cls
        cls_act = activations[mask_cls].mean(dim=0)  # [N]
        top3 = torch.topk(cls_act, min(3, N_PROBES)).indices.tolist()
        top3_vals = torch.topk(cls_act, min(3, N_PROBES)).values.tolist()
        name = ISAID5I_CATEGORIES.get(cls, str(cls))
        probes_str = ", ".join(f"p{p}({v:.3f})" for p, v in zip(top3, top3_vals))
        print(f"      {name:<20s}: {probes_str}")

    # 3c. Probe activation consistency across classes (coefficient of variation)
    print(f"\n  3c. Per-probe activation range across classes:")
    for p in range(N_PROBES):
        act_p = activations[:, p].numpy()
        per_class_mean = [act_p[class_labels == cls].mean() for cls in unique_classes]
        activation_range = max(per_class_mean) - min(per_class_mean)
        cv = np.std(per_class_mean) / max(np.mean(per_class_mean), 1e-8)
        bar = "█" * max(1, int(cv * 20))
        print(f"      Probe {p:>2}: range={activation_range:.4f}, CV={cv:.4f}  {bar}")

    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS 4: Attention Mask Analysis
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'=' * 72}")
    print("ANALYSIS 4: Attention Mask Degeneracy")
    print("  Q: Are attention masks degenerate (all-attend or none-attend)?")
    print("=" * 72)

    for layer_idx in range(N_LAYERS):
        frac_masked = np.mean([
            s["per_layer"][layer_idx]["attn_frac_masked"] for s in all_samples
        ])
        print(f"  Layer {layer_idx}: {frac_masked:.4f} pixels masked out (avg over samples)")
        if frac_masked < 0.01:
            print(f"    ⚠️  Almost NO masking → probes attend everywhere (not focused)")
        elif frac_masked > 0.99:
            print(f"    ⚠️  Almost ALL masked → probes attend nowhere (degenerate)")
        elif 0.1 < frac_masked < 0.9:
            print(f"    ✅ Reasonable masking range")

    # ═══════════════════════════════════════════════════════════════════
    # ANALYSIS 5: Probe Confidence Distribution
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'=' * 72}")
    print("ANALYSIS 5: Probe Confidence / Logit Distribution")
    print("  Q: Do probes have differentiated confidence, or are they all equal?")
    print("=" * 72)

    all_logits = torch.stack(
        [s["per_layer"][FINAL]["probe_logits"] for s in all_samples], dim=0)  # [S, N]
    mean_logits = all_logits.mean(dim=0)  # [N]
    # Softmax distribution
    softmax_dist = torch.softmax(all_logits, dim=1).mean(dim=0)  # [N]

    print(f"\n  Per-probe mean logit and softmax weight:")
    print(f"  {'Probe':<8} {'Mean Logit':>12} {'Softmax wt':>12} {'Bar'}")
    print(f"  {'-' * 50}")
    for p in range(N_PROBES):
        bar = "█" * max(1, int(softmax_dist[p] * 60 / softmax_dist.max().item()))
        print(f"  {p:<8} {mean_logits[p]:>12.4f} {softmax_dist[p]:>12.4f}  {bar}")

    entropy = -(softmax_dist * torch.log(softmax_dist + 1e-8)).sum()
    max_entropy = np.log(N_PROBES)
    print(f"\n  Softmax entropy: {entropy:.4f} / {max_entropy:.4f} (max)")
    if entropy < 1.0:
        print(f"  ⚠️  DIAGNOSIS: One probe dominates → effective single-probe SPG")
    elif entropy > max_entropy * 0.8:
        print(f"  ✅ DIAGNOSIS: Near-uniform probe weighting → all probes equally used")

    # ═══════════════════════════════════════════════════════════════════
    # DECISION TREE
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'─' * 72}")
    print("DECISION TREE:")
    print()
    print("  Analysis 1 (Spatial):")
    print(f"    Mean pairwise IoU = {mean_pairwise_iou:.4f}")
    if mean_pairwise_iou > 0.7:
        print(f"    → SPATIAL COLLAPSE: probes all look at the same region")
    else:
        print(f"    → Probes have spatial diversity")
    print()
    print("  Analysis 2 (Feature):")
    print(f"    cos(probe_i, probe_j) = {off_diag_q:.6f}")
    if off_diag_q > 0.95:
        print(f"    → FEATURE COLLAPSE: all probes encode the same representation")
    else:
        print(f"    → Probes have feature diversity")
    print()
    print("  Analysis 3 (Class Sensitivity):")
    print(f"    Mean F-score = {mean_f:.2f}, {n_sensitive}/{N_PROBES} class-sensitive")
    if n_sensitive == 0:
        print(f"    → SPG = OBJECTNESS GENERATOR (all probes encode foreground, not class)")
        print(f"    → This is the ROOT CAUSE of low mIoU despite good FB-IoU")
    elif n_sensitive <= 3:
        print(f"    → SPG = MOSTLY objectness, few semantic probes")
    else:
        print(f"    → SPG has meaningful class sensitivity")
    print()
    print("  Overall:")
    if n_sensitive == 0 and mean_pairwise_iou > 0.5:
        print(f"    → SPG is a pure Objectness Generator: spatially overlapped, semantically blind")
    elif n_sensitive == 0 and off_diag_q > 0.9:
        print(f"    → SPG probes collapsed to a single objectness pattern")
    elif n_sensitive == 0:
        print(f"    → SPG probes have spatial diversity but NO semantic content")
        print(f"    → Need: semantic supervision on per-probe masks (per-class probe assignment)")
    else:
        print(f"    → SPG has some class sensitivity, bottleneck likely elsewhere")


if __name__ == "__main__":
    main()
