"""
Support-Query 特征相似度分析 | Feature Similarity Analysis.
============================================================

验证 Stage 1 Domain Adapter 是否提升了特征的可匹配性（matching），
而非仅仅是分类性（classification）。

核心指标 | Core Metric:
    intra-class cos-sim / inter-class cos-sim 比值。
    比值越高 → 同类 support-query 越聚拢，不同类越分散 → 特征更适合 few-shot matching。

用法 | Usage:
    # 对比 raw SAM vs Stage1 adapter 特征
    python tools/analysis/feature_similarity.py --fold 1 --k-shot 5

    # 指定 Stage1 checkpoint
    python tools/analysis/feature_similarity.py --fold 1 --stage1-ckpt runs/stage1_fold1_seed42/best_adapter.pt
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
from adasam.datasets.isaid_5i import ISAID5iDataset, ISAID5I_CATEGORIES
from adasam.utils import set_seed
from adasam.utils.transforms import preprocess_image


def extract_features(
    backbone: MobileSAMBackbone,
    adapter: CATAdapter | None,
    dataset: ISAID5iDataset,
    indices: list[int],
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Extract adapter features for each tile (averaged over FG region)."""
    features: dict[int, torch.Tensor] = {}  # {class_id: stacked features}
    feat_list: dict[int, list[torch.Tensor]] = {}

    for idx in tqdm(indices, desc="extract features"):
        sample = dataset[idx]
        x, _ = preprocess_image(sample["image"])
        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            emb = backbone(x)["image_embedding"]  # [1, 256, 64, 64]
            if adapter is not None:
                emb = adapter(emb)

        # For each visible class, get FG mask → mean-pool over FG region
        for cls_id in dataset.visible_classes():
            mask = dataset.get_class_mask(idx, cls_id)
            if mask is None or mask.sum() < 10:  # skip tiny/no regions
                continue

            # Resize mask from 256² to 64² (feature map size)
            mask_64 = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(64, 64), mode="nearest",
            ).squeeze() > 0.5  # [64, 64]

            # Mean-pool over FG region → [256]
            feat_fg = emb[0, :, mask_64].mean(dim=1)  # [256]
            feat_fg = F.normalize(feat_fg, dim=0)       # L2 normalize

            feat_list.setdefault(cls_id, []).append(feat_fg.cpu())

    # Stack per class
    for cls_id, feats in feat_list.items():
        features[cls_id] = torch.stack(feats, dim=0)  # [N, 256]

    return features


def compute_similarity_stats(features: dict[int, torch.Tensor]) -> dict:
    """Compute intra-class and inter-class cosine similarity statistics."""
    intra_sims = []
    inter_sims = []

    class_ids = sorted(features.keys())

    for i, c in enumerate(class_ids):
        feats_c = features[c]  # [N_c, 256]
        n_c = feats_c.shape[0]

        # Intra-class: all pairs within same class
        if n_c >= 2:
            sim_matrix = feats_c @ feats_c.T  # [N_c, N_c]
            # Upper triangle excluding diagonal
            triu = sim_matrix[np.triu_indices(n_c, k=1)]
            intra_sims.extend(triu.tolist())

        # Inter-class: against all other classes
        for j, c2 in enumerate(class_ids):
            if j <= i:
                continue
            feats_c2 = features[c2]  # [N_c2, 256]
            # All cross-class pairs
            cross_sim = (feats_c @ feats_c2.T).flatten()  # [N_c * N_c2]
            inter_sims.extend(cross_sim.tolist())

    intra = np.array(intra_sims)
    inter = np.array(inter_sims)

    # Separability ratio: how much more similar are same-class vs diff-class features
    separability = intra.mean() / max(inter.mean(), 1e-8) if len(intra) > 0 else 0.0

    return {
        "intra_mean": float(intra.mean()) if len(intra) > 0 else 0.0,
        "intra_std": float(intra.std()) if len(intra) > 0 else 0.0,
        "inter_mean": float(inter.mean()) if len(inter) > 0 else 0.0,
        "inter_std": float(inter.std()) if len(inter) > 0 else 0.0,
        "separability": float(separability),
        "intra_n": len(intra),
        "inter_n": len(inter),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Support-Query Feature Similarity Analysis")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--stage1-ckpt", default=None,
                   help="path to Stage 1 adapter checkpoint. Omit for raw SAM.")
    p.add_argument("--data-root", default=None)
    p.add_argument("--weights", default=None,
                   help="MobileSAM weights (default: weights/mobile_sam.pt)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=200,
                   help="max tiles per split for speed")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    data_root = args.data_root or str(_REPO_ROOT / "data" / "iSAID-5i")
    weights = args.weights or str(_REPO_ROOT / "weights" / "mobile_sam.pt")

    # ── Backbone ──
    sam = build_mobile_sam(weights, "vit_t", device)
    backbone = MobileSAMBackbone(sam.image_encoder, sam.image_encoder.img_size).to(device)
    backbone.eval()

    # ── Adapter (optional) ──
    adapter = None
    adapter_name = "raw_SAM"
    if args.stage1_ckpt:
        ckpt = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
        adapter_state = ckpt.get("adapter")
        adapter_cfg = ckpt.get("config", {}).get("adapter", {})
        adapter = CATAdapter(
            dim=256, bottleneck=int(adapter_cfg.get("bottleneck", 64)),
        ).to(device)
        adapter.load_state_dict(adapter_state)
        adapter.eval()
        adapter_name = f"stage1_ep{ckpt.get('epoch','?')}"
        print(f"[adapter] loaded: {args.stage1_ckpt}")

    # ── Dataset (train split, base classes) ──
    ds = ISAID5iDataset(root=data_root, fold=args.fold, split="train", mode="base")
    classes = sorted(ds.visible_classes())
    print(f"\n[dataset] fold={args.fold} classes={classes} tiles={len(ds)}")

    # Sample tiles for speed
    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    sampled = indices[:args.max_samples]

    # ── Extract features ──
    print(f"\n=== Feature Extraction: {adapter_name} ===")
    features = extract_features(backbone, adapter, ds, sampled, device)

    # Per-class tile count
    print("\nPer-class feature vectors:")
    for cls_id in sorted(features.keys()):
        name = ISAID5I_CATEGORIES.get(cls_id, "?")
        print(f"  class {cls_id:>2d} ({name:<20s}): {features[cls_id].shape[0]} vectors")

    # ── Similarity ──
    stats = compute_similarity_stats(features)
    print(f"\n=== Similarity Stats: {adapter_name} ===")
    print(f"  intra-class cos-sim:  {stats['intra_mean']:.4f} ± {stats['intra_std']:.4f}  (n={stats['intra_n']})")
    print(f"  inter-class cos-sim:  {stats['inter_mean']:.4f} ± {stats['inter_std']:.4f}  (n={stats['inter_n']})")
    print(f"  separability ratio:   {stats['separability']:.2f}x  (intra/inter, higher=better for matching)")

    # Interpretation
    ratio = stats['separability']
    if ratio < 1.1:
        verdict = "⚠️  FEATURES NOT SEPARABLE — same-class pairs are barely more similar than different-class pairs"
    elif ratio < 1.5:
        verdict = "⚡ marginal separation — matching will be noisy"
    elif ratio < 2.5:
        verdict = "✓  moderate separation — usable for few-shot matching"
    else:
        verdict = "✓✓ strong separation — good for few-shot matching"
    print(f"\n  → {verdict}")

    # ── If we have both raw and adapted, print comparison hints ──
    if adapter is not None:
        print(f"\n=== How to run baseline comparison ===")
        print(f"  python tools/analysis/feature_similarity.py --fold {args.fold} "
              f"--data-root {data_root} --weights {weights}")


if __name__ == "__main__":
    main()
