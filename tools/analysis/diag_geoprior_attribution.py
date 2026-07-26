"""
GeoPrior 灵敏度归因 | GeoPrior Sensitivity Attribution.
========================================================

诊断 GeoPrior 到底是 "query encoder" 还是 "support-conditioned projector":

  Exp 1: 固定 query，换 support → GeoPrior 输出变化量
  Exp 2: 固定 support，换 query → GeoPrior 输出变化量

如果 Exp2 变化量 >> Exp1 变化量:
  GeoPrior 已退化为 query encoder，support 只是微弱条件。
  根因确认：GeoPrior 没有把 prototype 类别信息传递下去。

用法 | Usage:
    python tools/analysis/diag_geoprior_attribution.py \
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

def masked_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Per-channel mean cosine between two [1, C, H, W] tensors."""
    a0 = a[0] if a.ndim == 4 else a
    b0 = b[0] if b.ndim == 4 else b
    C = a0.shape[0]
    af = a0.reshape(C, -1).float()
    bf = b0.reshape(C, -1).float()
    an = F.normalize(af, dim=1)
    bn = F.normalize(bf, dim=1)
    return float((an * bn).sum(dim=1).mean().item())


def build_support_for_class(
    dataset, class_id: int, k_shot: int, device: torch.device,
    backbone: MobileSAMBackbone, adapter,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Build (support_features, support_masks) for a class."""
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


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GeoPrior Sensitivity Attribution")
    parser.add_argument("--stage2-ckpt", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=30,
                        help="Number of (query, support_class) pairs to test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    print(f"Device: {device}")

    # ── Load ──
    ckpt_path = Path(args.stage2_ckpt)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

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

    if model.geometric_prior is None:
        print("ERROR: GeometricPrior not enabled in this checkpoint")
        sys.exit(1)

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

    data_root = str(_REPO_ROOT / args.data_root) if not Path(args.data_root).is_absolute() \
        else args.data_root
    dataset = ISAID5iDataset(
        root=data_root, fold=args.fold, split="val", mode="base")
    val_classes = sorted(dataset.visible_classes())

    # ── Sample (query_tile, support_class) pairs ──
    tile_pool = list(range(len(dataset)))
    rng.shuffle(tile_pool)
    # Pick tiles that have at least one GT class present
    valid_tiles = [t for t in tile_pool[:200]
                   if len(dataset[t]["regions"]) > 0][:args.n_samples]

    # For each tile, pick one class present as "reference support"
    pairs = []
    for tile_idx in valid_tiles:
        sample = dataset[tile_idx]
        classes_present = set(r["category_id"] for r in sample["regions"]
                              if r["category_id"] in val_classes)
        if not classes_present:
            continue
        ref_cls = rng.choice(list(classes_present))
        # Pick a second class if available
        other_classes = [c for c in val_classes if c != ref_cls
                         and len(dataset.class_to_tiles(c)) >= args.k_shot]
        if not other_classes:
            continue
        other_cls = rng.choice(other_classes)
        pairs.append((tile_idx, ref_cls, other_cls))

    print(f"Testing {len(pairs)} (query, support_A, support_B) triplets\n")

    # ═══════════════════════════════════════════════════════════════
    # Exp 1: Fix query, swap support → GeoPrior sensitivity to support
    # ═══════════════════════════════════════════════════════════════

    # For each query, compute GeoPrior with 2 different supports → compare
    sup_swap_cosines = []  # cos between GeoPrior(support_A) and GeoPrior(support_B)

    # ═══════════════════════════════════════════════════════════════
    # Exp 2: Fix support, swap query → GeoPrior sensitivity to query
    # ═══════════════════════════════════════════════════════════════

    # For each support, compute GeoPrior on 2 different queries → compare
    qry_swap_cosines = []

    # We need paired data: (query_A, query_B, support) and (query, support_A, support_B)
    # Use every pair as both: tile_i ↔ tile_j with the same support class, and
    # tile ↔ tile with different supports.

    # ── Exp 1: fix query, swap support ──
    print("Exp 1: Fix query, swap support ...")
    for tile_idx, cls_a, cls_b in tqdm(pairs, desc="sup-swap"):
        sample = dataset[tile_idx]
        x, _ = preprocess_image(sample["image"])
        with torch.no_grad():
            q_emb = backbone(x.unsqueeze(0).to(device))["image_embedding"]
            if adapter is not None:
                q_emb = adapter(q_emb)

        # Support A
        sup_a = build_support_for_class(
            dataset, cls_a, args.k_shot, device, backbone, adapter, rng)
        if sup_a is None:
            continue
        sup_feat_a, sup_mask_a = sup_a
        sm_a = model.support_encoder(sup_feat_a, sup_mask_a)
        gp_a = model.geometric_prior(q_emb, sm_a)

        # Support B
        sup_b = build_support_for_class(
            dataset, cls_b, args.k_shot, device, backbone, adapter, rng)
        if sup_b is None:
            continue
        sup_feat_b, sup_mask_b = sup_b
        sm_b = model.support_encoder(sup_feat_b, sup_mask_b)
        gp_b = model.geometric_prior(q_emb, sm_b)

        cos_ab = masked_cosine(gp_a, gp_b)
        sup_swap_cosines.append(cos_ab)

    # ── Exp 2: fix support, swap query ──
    print("Exp 2: Fix support, swap query ...")
    # Use consecutive pairs of tiles with same support class
    for i in range(0, len(pairs) - 1, 2):
        tile_a, cls_a, _ = pairs[i]
        tile_b, cls_b, _ = pairs[i + 1]
        # Use cls_a as the fixed support class
        sup_data = build_support_for_class(
            dataset, cls_a, args.k_shot, device, backbone, adapter, rng)
        if sup_data is None:
            continue
        sup_feat, sup_mask = sup_data
        sm = model.support_encoder(sup_feat, sup_mask)

        # Query A
        sample_a = dataset[tile_a]
        x_a, _ = preprocess_image(sample_a["image"])
        with torch.no_grad():
            q_a = backbone(x_a.unsqueeze(0).to(device))["image_embedding"]
            if adapter is not None:
                q_a = adapter(q_a)
        gp_a = model.geometric_prior(q_a, sm)

        # Query B
        sample_b = dataset[tile_b]
        x_b, _ = preprocess_image(sample_b["image"])
        with torch.no_grad():
            q_b = backbone(x_b.unsqueeze(0).to(device))["image_embedding"]
            if adapter is not None:
                q_b = adapter(q_b)
        gp_b = model.geometric_prior(q_b, sm)

        cos_ab = masked_cosine(gp_a, gp_b)
        qry_swap_cosines.append(cos_ab)

    # ═══════════════════════════════════════════════════════════════
    # Report
    # ═══════════════════════════════════════════════════════════════

    mean_sup = np.mean(sup_swap_cosines) if sup_swap_cosines else 0.0
    std_sup = np.std(sup_swap_cosines) if sup_swap_cosines else 0.0
    mean_qry = np.mean(qry_swap_cosines) if qry_swap_cosines else 0.0
    std_qry = np.std(qry_swap_cosines) if qry_swap_cosines else 0.0

    # Sensitivity = 1 - cos (higher = more sensitive / more change when swapped)
    sup_sensitivity = 1.0 - mean_sup  # how much GeoPrior changes when support changes
    qry_sensitivity = 1.0 - mean_qry  # how much GeoPrior changes when query changes

    dominance_ratio = qry_sensitivity / max(sup_sensitivity, 1e-8)

    print(f"\n{'=' * 72}")
    print("GeoPrior Sensitivity Attribution")
    print("=" * 72)

    print(f"\n  Exp 1: Fix query, swap support class")
    print(f"    N={len(sup_swap_cosines)}  mean_cos={mean_sup:.4f}  std={std_sup:.4f}")
    print(f"    Sensitivity (1-cos) = {sup_sensitivity:.4f}")
    if sup_sensitivity < 0.1:
        print(f"    ❌ GeoPrior IGNORES support class — output barely changes")
    elif sup_sensitivity < 0.3:
        print(f"    ⚠️  GeoPrior has WEAK response to support class")
    else:
        print(f"    ✅ GeoPrior IS sensitive to support class")

    print(f"\n  Exp 2: Fix support, swap query image")
    print(f"    N={len(qry_swap_cosines)}  mean_cos={mean_qry:.4f}  std={std_qry:.4f}")
    print(f"    Sensitivity (1-cos) = {qry_sensitivity:.4f}")
    if qry_sensitivity < 0.1:
        print(f"    Query change has LITTLE effect (unusual)")
    elif qry_sensitivity < 0.5:
        print(f"    Query change has MODERATE effect")
    else:
        print(f"    ❌ GeoPrior DOMINATED by query — support lost")

    print(f"\n{'─' * 60}")
    print(f"  Dominance Ratio: query_sensitivity / support_sensitivity")
    print(f"    = {qry_sensitivity:.4f} / {sup_sensitivity:.4f}")
    print(f"    = {dominance_ratio:.2f}×")

    if dominance_ratio > 5:
        print(f"\n  🔴 GeoPrior is effectively a QUERY ENCODER")
        print(f"     It's {dominance_ratio:.0f}× more sensitive to changing the query")
        print(f"     than to changing the support class.")
        print(f"     Support provides almost NO conditioning signal.")
    elif dominance_ratio > 2:
        print(f"\n  🟡 GeoPrior is QUERY-DOMINATED")
        print(f"     Query changes matter {dominance_ratio:.0f}× more than support changes.")
        print(f"     Class information from support is heavily diluted.")
    elif dominance_ratio > 0.5:
        print(f"\n  🟢 GeoPrior is BALANCED")
    else:
        print(f"\n  ✅ GeoPrior IS support-conditioned")

    # ── Also check: within-class support consistency ──
    # For the same class, different support samples → should produce similar GeoPrior
    print(f"\n{'=' * 72}")
    print("Bonus: Within-Class Support Consistency")
    print("=" * 72)

    within_cls_cosines = []
    for tile_idx, cls_ref, _ in pairs[:15]:
        sample = dataset[tile_idx]
        x, _ = preprocess_image(sample["image"])
        with torch.no_grad():
            q_emb = backbone(x.unsqueeze(0).to(device))["image_embedding"]
            if adapter is not None:
                q_emb = adapter(q_emb)

        # Two different support sets for the same class
        sup_1 = build_support_for_class(
            dataset, cls_ref, args.k_shot, device, backbone, adapter, rng)
        sup_2 = build_support_for_class(
            dataset, cls_ref, args.k_shot, device, backbone, adapter, rng)
        if sup_1 is None or sup_2 is None:
            continue

        sm1 = model.support_encoder(sup_1[0], sup_1[1])
        sm2 = model.support_encoder(sup_2[0], sup_2[1])
        gp1 = model.geometric_prior(q_emb, sm1)
        gp2 = model.geometric_prior(q_emb, sm2)

        within_cls_cosines.append(masked_cosine(gp1, gp2))

    mean_within = np.mean(within_cls_cosines) if within_cls_cosines else 0.0
    print(f"  Same class, different support samples: mean_cos={mean_within:.4f}")
    print(f"  Cross-class, same query:               mean_cos={mean_sup:.4f}")
    print(f"  Gap (within - cross) = {mean_within - mean_sup:+.4f}")
    if mean_within - mean_sup < 0.02:
        print(f"  ❌ GeoPrior cannot even distinguish same-class from different-class support")
    elif mean_within - mean_sup < 0.1:
        print(f"  ⚠️  Weak within-class consistency — support sampling noise may dominate")
    else:
        print(f"  ✅ GeoPrior has some within-class consistency")


if __name__ == "__main__":
    main()
