"""
Prototype 类别检索诊断 | Prototype Class Retrieval Diagnostic.
===============================================================

绕过 GeometricPrior / SPG / PromptFusion / SAM Decoder，直接用 support prototype
在 query feature map 上做最近邻检索。回答一个根本问题：

    prototype 本身是否编码了正确的类别信息？

实验：
  1. Per-class prototype → 对所有 query tiles 做最近邻检索
  2. 统计 Top-1/5/10/50 命中正确 GT 的比率
  3. Prototype 间的距离矩阵（哪些类别容易混淆）
  4. 对 storage_tank / roundabout / plane 做重点可视化分析

如果 prototype 检索就失败 → 根因在 Stage1 feature / support 数据质量
如果 prototype 检索很好但 mask 不行 → 根因在 Prompt 生成链或 Mask Decoder

用法 | Usage:
    python tools/analysis/diag_prototype_retrieval.py \
        --stage2-ckpt runs/stage2_fold1_k5_seed42/last_model.pt \
        --data-root data/iSAID-5i --fold 1 --k-shot 5
"""

from __future__ import annotations

import argparse
import math
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
from adasam.utils import set_seed
from adasam.utils.transforms import preprocess_image, resize_mask


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def build_support_for_class(
    dataset, class_id: int, k_shot: int, device: torch.device,
    backbone: MobileSAMBackbone, adapter,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Build (support_features, support_masks) for a class (FSS protocol)."""
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


def compute_prototype(features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Masked-mean prototype: support FG pixels → single [C] vector."""
    masked = features * masks.unsqueeze(1)      # [K, C, H, W]
    fg_sum = masked.sum(dim=(0, 2, 3))          # [C]
    fg_count = masks.sum() + 1e-8
    return fg_sum / fg_count                     # [C]


def pixel_classification_accuracy(
    query_features: torch.Tensor,    # [1, C, H, W]
    prototype: torch.Tensor,         # [C]
    gt_mask: np.ndarray,             # [H, W] bool
    top_k: int = 1,
) -> tuple[float, float, float]:
    """Classify each FG pixel by cosine similarity to prototype.

    For each pixel in the query feature map, compute cosine similarity to prototype.
    Treat it as a "retrieval" task: the prototype "retrieves" the most similar pixels.

    Returns:
        (precision, recall, hit_rate)
        precision: fraction of Top-K pixels that fall on GT
        recall: fraction of GT pixels covered by Top-K
        hit_rate: fraction of query tiles where Top-1 pixel hits GT
    """
    C, H, W = query_features.shape[1:]
    flat = query_features.reshape(1, C, -1)            # [1, C, N]
    flat_norm = F.normalize(flat, dim=1)               # [1, C, N]
    proto_norm = F.normalize(prototype, dim=0).reshape(1, C, 1)  # [1, C, 1]

    sim = (flat_norm * proto_norm).sum(dim=1)           # [1, N]
    sim_map = sim.reshape(1, H, W)                 # [1, H, W]

    # Top-K pixels
    N = H * W
    k = min(top_k, N)
    _, top_indices = sim.view(-1).topk(k)
    top_mask = torch.zeros(N, dtype=torch.bool, device=sim.device)
    top_mask[top_indices] = True
    top_mask = top_mask.reshape(H, W).cpu().numpy()

    # Resize GT from original size to feature grid size
    gt_resized = resize_mask(torch.from_numpy(gt_mask.astype(float)), (H, W)).numpy().astype(bool)

    hits = float((top_mask & gt_resized).sum())
    precision = hits / k
    recall = hits / max(float(gt_resized.sum()), 1.0)
    hit_rate = 1.0 if hits > 0 else 0.0

    return precision, recall, hit_rate


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Prototype Class Retrieval Diagnostic"
    )
    parser.add_argument("--stage2-ckpt", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--n-queries", type=int, default=50,
                        help="Number of random query tiles per class")
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
    print(f"  epoch={ckpt.get('epoch', '?')}  fold={ckpt.get('fold', '?')}")

    # ── Build backbone ──
    weights_path = str(_REPO_ROOT / cfg.get("backbone", {}).get(
        "checkpoint", "weights/mobile_sam.pt"))
    sam = build_mobile_sam(
        weights_path, cfg.get("backbone", {}).get("model_type", "vit_t"), device)
    backbone = MobileSAMBackbone(
        sam.image_encoder, sam.image_encoder.img_size).to(device)

    # ── Load adapter ──
    adapter_state = ckpt.get("cat_adapter")
    if adapter_state is not None:
        adapter_cfg = ckpt.get("config", {}).get("adapter", {})
        adapter = CATAdapter(
            dim=256,
            bottleneck=int(adapter_cfg.get("bottleneck", 64)),
        ).to(device)
        adapter.load_state_dict(adapter_state)
        adapter.eval()
        for p in adapter.parameters():
            p.requires_grad_(False)
        print("  adapter: loaded + frozen")
    else:
        adapter = None
        print("  adapter: NONE (raw SAM features)")

    # ── Dataset ──
    data_root = str(_REPO_ROOT / args.data_root) if not Path(args.data_root).is_absolute() \
        else args.data_root
    dataset = ISAID5iDataset(
        root=data_root, fold=args.fold, split="val", mode="base")
    val_classes = sorted(dataset.visible_classes())
    print(f"Val classes ({len(val_classes)}): {val_classes}\n")

    # ═══════════════════════════════════════════════════════════════
    # EXP 1: Prototype × Prototype Distance Matrix
    # ═══════════════════════════════════════════════════════════════

    print("=" * 72)
    print("EXP 1: Prototype Cross-Class Distance Matrix")
    print("  (cosine similarity × 100, higher = more similar/confusable)")
    print("=" * 72)

    prototypes = {}  # class_id → [C] prototype
    for cls in val_classes:
        sup_data = build_support_for_class(
            dataset, cls, args.k_shot, device, backbone, adapter, rng
        )
        if sup_data is not None:
            sup_feat, sup_mask = sup_data
            prototypes[cls] = compute_prototype(sup_feat, sup_mask)
        else:
            prototypes[cls] = None

    n_cls = len(val_classes)
    proto_matrix = np.zeros((n_cls, n_cls))
    for i, ci in enumerate(val_classes):
        for j, cj in enumerate(val_classes):
            pi, pj = prototypes.get(ci), prototypes.get(cj)
            if pi is not None and pj is not None:
                sim = float(F.cosine_similarity(
                    pi.unsqueeze(0), pj.unsqueeze(0)).item())
                proto_matrix[i, j] = sim * 100  # scale to percentage
            else:
                proto_matrix[i, j] = np.nan

    # Print
    col_w = 8
    header = f"{'':>10s}"
    for c in val_classes:
        name = ISAID5I_CATEGORIES.get(c, str(c))[:col_w]
        header += f"  {name:>{col_w}s}"
    print(header)
    for i, ci in enumerate(val_classes):
        name = ISAID5I_CATEGORIES.get(ci, str(ci))
        row = f"  {name:>8s}"
        for j in range(n_cls):
            val = proto_matrix[i, j]
            marker = ""
            if i != j and not math.isnan(val) and val > 90:
                marker = "⚠️"
            row += f"  {val:{col_w}.1f}{marker}"
        print(row)

    # Find confusing pairs
    print(f"\n{'─' * 60}")
    print("Highly confusable prototype pairs (cos > 0.90):")
    confusing = []
    for i in range(n_cls):
        for j in range(i + 1, n_cls):
            if proto_matrix[i, j] > 90:
                ci, cj = val_classes[i], val_classes[j]
                name_i = ISAID5I_CATEGORIES.get(ci, str(ci))
                name_j = ISAID5I_CATEGORIES.get(cj, str(cj))
                confusing.append((ci, cj, proto_matrix[i, j]))
                print(f"  {name_i} ↔ {name_j}: cos={proto_matrix[i, j]:.1f}")
    if not confusing:
        print("  (none — all prototype pairs well-separated)")

    # ═══════════════════════════════════════════════════════════════
    # EXP 2: Prototype → Pixel-Level Retrieval
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 72}")
    print("EXP 2: Prototype → Pixel Retrieval (cosine similarity)")
    print("  For each class, retrieve Top-K pixels from query tiles")
    print("=" * 72)

    # Collect query tiles per class
    class_tiles = defaultdict(list)
    for idx in range(len(dataset)):
        for cls in val_classes:
            gt = dataset.get_class_mask(idx, cls)
            if gt is not None and gt.sum() > 100:  # enough FG pixels
                class_tiles[cls].append(idx)

    # Per-class retrieval stats
    retrieval_stats: dict[int, dict[str, float]] = {}

    for cls in val_classes:
        proto = prototypes.get(cls)
        if proto is None:
            retrieval_stats[cls] = {"precision@1": 0, "precision@10": 0,
                                     "recall@10": 0, "hit_rate": 0, "n_tested": 0}
            continue

        tiles = class_tiles[cls]
        if len(tiles) > args.n_queries:
            tiles = rng.sample(tiles, args.n_queries)

        prec_1_list, prec_10_list, rec_10_list, hit_list = [], [], [], []
        for idx in tiles:
            sample = dataset[idx]
            gt = dataset.get_class_mask(idx, cls)
            if gt is None:
                continue
            gt_np = gt.numpy().astype(bool)

            x, _ = preprocess_image(sample["image"])
            with torch.no_grad():
                q_emb = backbone(x.unsqueeze(0).to(device))["image_embedding"]
                if adapter is not None:
                    q_emb = adapter(q_emb)

            # Retrieve Top-1 and Top-10
            p1, _, h1 = pixel_classification_accuracy(q_emb, proto, gt_np, top_k=1)
            p10, r10, _ = pixel_classification_accuracy(q_emb, proto, gt_np, top_k=10)

            prec_1_list.append(p1)
            prec_10_list.append(p10)
            rec_10_list.append(r10)
            hit_list.append(h1)

        retrieval_stats[cls] = {
            "precision@1": np.mean(prec_1_list) if prec_1_list else 0.0,
            "precision@10": np.mean(prec_10_list) if prec_10_list else 0.0,
            "recall@10": np.mean(rec_10_list) if rec_10_list else 0.0,
            "hit_rate": np.mean(hit_list) if hit_list else 0.0,
            "n_tested": len(prec_1_list),
        }

    # Print retrieval stats
    print(f"\n{'Class':>20s}  {'n_tiles':>7s}  {'P@1':>7s}  {'P@10':>7s}  "
          f"{'R@10':>7s}  {'HitRate':>7s}  {'Status':>12s}")
    print("-" * 80)
    for cls in val_classes:
        name = ISAID5I_CATEGORIES.get(cls, str(cls))
        s = retrieval_stats[cls]
        n = s["n_tested"]
        p1 = s["precision@1"]
        p10 = s["precision@10"]
        r10 = s["recall@10"]
        hr = s["hit_rate"]

        if n == 0:
            status = "NO DATA"
        elif p1 >= 0.3:
            status = "✅ GOOD"
        elif p1 >= 0.1:
            status = "⚠️  WEAK"
        else:
            status = "❌ FAIL"

        print(f"  {name:>20s}  {n:>7d}  {p1:>6.1%}  {p10:>6.1%}  "
              f"{r10:>6.1%}  {hr:>6.0%}  {status:>12s}")

    # ═══════════════════════════════════════════════════════════════
    # EXP 3: Focus on problematic classes
    # ═══════════════════════════════════════════════════════════════

    problem_classes = [c for c in val_classes
                       if retrieval_stats[c].get("precision@1", 0) < 0.1]

    if problem_classes:
        print(f"\n{'=' * 72}")
        print("EXP 3: Problem Class Deep Dive")
        print(f"  Classes with P@1 < 10%: "
              f"{[ISAID5I_CATEGORIES.get(c, str(c)) for c in problem_classes]}")
        print("=" * 72)

        for cls in problem_classes:
            name = ISAID5I_CATEGORIES.get(cls, str(cls))
            proto = prototypes.get(cls)
            if proto is None:
                print(f"\n  {name}: prototype unavailable (no support data)")
                continue

            # Check: which other prototype is nearest to ours?
            proto_distances = []
            for other_cls in val_classes:
                if other_cls == cls:
                    continue
                other_proto = prototypes.get(other_cls)
                if other_proto is None:
                    continue
                sim = float(F.cosine_similarity(
                    proto.unsqueeze(0), other_proto.unsqueeze(0)).item())
                proto_distances.append((other_cls, sim))
            proto_distances.sort(key=lambda x: -x[1])

            print(f"\n  {name} (cls={cls}):")
            print(f"    Top-5 nearest OTHER prototypes:")
            for other_cls, sim in proto_distances[:5]:
                other_name = ISAID5I_CATEGORIES.get(other_cls, str(other_cls))
                print(f"      {other_name:<20s}  cos={sim:.4f}")

            # Check: what does the prototype activate on a query tile?
            tiles = class_tiles.get(cls, [])
            if tiles:
                tile_idx = rng.choice(tiles)
                sample = dataset[tile_idx]
                print(f"    Query tile {tile_idx}: GT classes present:")
                for gc in val_classes:
                    gt = dataset.get_class_mask(tile_idx, gc)
                    if gt is not None and gt.sum() > 50:
                        gc_name = ISAID5I_CATEGORIES.get(gc, str(gc))
                        print(f"      {gc_name:<20s}: {gt.sum():.0f} px")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print("=" * 72)

    n_good = sum(1 for cls in val_classes
                 if retrieval_stats[cls].get("precision@1", 0) >= 0.3)
    n_weak = sum(1 for cls in val_classes
                 if 0.1 <= retrieval_stats[cls].get("precision@1", 0) < 0.3)
    n_fail = sum(1 for cls in val_classes
                 if retrieval_stats[cls].get("precision@1", 0) < 0.1)

    mean_p1 = np.mean([retrieval_stats[c]["precision@1"] for c in val_classes])
    mean_p10 = np.mean([retrieval_stats[c]["precision@10"] for c in val_classes])

    print(f"  Prototype retrieval (no SPG/GeoPrior/Decoder):")
    print(f"    Mean P@1:         {mean_p1:.1%}")
    print(f"    Mean P@10:        {mean_p10:.1%}")
    print(f"    Good classes:     {n_good}/{len(val_classes)}")
    print(f"    Weak classes:     {n_weak}/{len(val_classes)}")
    print(f"    Failed classes:   {n_fail}/{len(val_classes)}")

    # Compare with confusion matrix results (from diag_confusion.py)
    print(f"\n  Prototype pairwise cos range: "
          f"{proto_matrix.min():.1f} – {proto_matrix.max():.1f}")

    if n_fail >= 3:
        print(f"\n  ❌ {n_fail} classes have nearly useless prototypes.")
        print(f"     These classes need better Stage1 features or more support data.")
        print(f"     Fixing prototype quality is PREREQUISITE to any downstream fix.")
    elif mean_p1 < 0.3:
        print(f"\n  ⚠️  Prototypes have weak but non-zero class signal.")
        print(f"     Downstream modules (SPG/GeoPrior) may amplify or lose this signal.")
    else:
        print(f"\n  ✅ Prototypes are strong — problem is downstream.")


if __name__ == "__main__":
    main()
