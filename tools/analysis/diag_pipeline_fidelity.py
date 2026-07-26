"""
Prompt 链信息保真度诊断 | Pipeline Fidelity Diagnostic.
========================================================

精确定位 GeoPrior → SPG → PromptFusion 中哪一步削弱了类别判别力。

实验：固定 query，用不同 support 类别 → 捕获每层中间输出 → 计算跨类别余弦相似度。

如果某一步跨类 cos 突然上升，说明那一步在"洗掉"类别信息。

用法 | Usage:
    python tools/analysis/diag_pipeline_fidelity.py \
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

def cosine_between_tensors(a: torch.Tensor, b: torch.Tensor) -> float:
    """Per-channel mean cosine similarity between two [1, C, H, W] tensors."""
    if a.ndim == 4:
        a = a[0]
        b = b[0]
    C = a.shape[0]
    a_flat = a.reshape(C, -1).float()
    b_flat = b.reshape(C, -1).float()
    a_n = F.normalize(a_flat, dim=1)
    b_n = F.normalize(b_flat, dim=1)
    return float((a_n * b_n).sum(dim=1).mean().item())


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


def compute_prototype(features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Masked-mean prototype [C]."""
    masked = features * masks.unsqueeze(1)
    fg_sum = masked.sum(dim=(0, 2, 3))
    fg_count = masks.sum() + 1e-8
    return fg_sum / fg_count


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Fidelity Diagnostic"
    )
    parser.add_argument("--stage2-ckpt", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--n-queries", type=int, default=10,
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

    # ── Build model ──
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
    else:
        adapter = None

    # ── Dataset ──
    data_root = str(_REPO_ROOT / args.data_root) if not Path(args.data_root).is_absolute() \
        else args.data_root
    dataset = ISAID5iDataset(
        root=data_root, fold=args.fold, split="val", mode="base")
    val_classes = sorted(dataset.visible_classes())

    print(f"Val classes: {val_classes}")

    # ── Find multi-class query tiles ──
    tile_scores = []
    for idx in range(len(dataset)):
        classes_present = []
        for cls in val_classes:
            gt = dataset.get_class_mask(idx, cls)
            if gt is not None and gt.sum() > 50:
                classes_present.append(cls)
        if len(classes_present) >= 2:
            tile_scores.append((idx, len(classes_present), classes_present))

    tile_scores.sort(key=lambda x: -x[1])
    n_queries = min(args.n_queries, len(tile_scores))
    selected = tile_scores[:n_queries]
    print(f"Multi-class tiles: {len(tile_scores)}, using {n_queries}")

    # ═══════════════════════════════════════════════════════════════
    # Main experiment: Cross-class cosine at each pipeline stage
    # ═══════════════════════════════════════════════════════════════

    # Collect per-stage pairwise cosines
    # stage_name → [(cos, cls_a_name, cls_b_name), ...]
    stage_data: dict[str, list[tuple[float, str, str]]] = defaultdict(list)

    # Also collect per-stage per-class "class vectors" for separation analysis
    # stage_name → {cls → [vec_C]}

    print(f"\n{'=' * 72}")
    print("Pipeline Fidelity: Cross-Class Cosine at Each Stage")
    print("  (lower cos = more class-specific = better)")
    print("=" * 72)

    for qi, (tile_idx, n_cls_on_tile, cls_list) in enumerate(
            tqdm(selected, desc="queries")):
        sample = dataset[tile_idx]

        # Embed query once
        x, _ = preprocess_image(sample["image"])
        with torch.no_grad():
            q_emb = backbone(x.unsqueeze(0).to(device))["image_embedding"]
            if adapter is not None:
                q_emb = adapter(q_emb)

        dense_pe = model.sam_decoder.prompt_encoder.get_dense_pe()

        # Build support and compute intermediates for each class
        per_class = {}  # cls → {stage: tensor}
        for cls in cls_list[:4]:  # max 4 classes per tile
            sup_data = build_support_for_class(
                dataset, cls, args.k_shot, device, backbone, adapter, rng
            )
            if sup_data is None:
                continue
            sup_feat, sup_mask = sup_data

            # ── Run each stage individually ──
            intermediates = {}

            # Stage: Prototype (baseline)
            proto = compute_prototype(sup_feat, sup_mask)
            intermediates["prototype"] = proto

            # Stage: SupportMemory
            support_memory = model.support_encoder(sup_feat, sup_mask)
            intermediates["support_memory"] = support_memory.mean(dim=0)  # [C] pooled

            # Stage: GeometricPrior
            if model.geometric_prior is not None:
                gp = model.geometric_prior(q_emb, support_memory)
                intermediates["geometric_prior"] = gp
            else:
                intermediates["geometric_prior"] = None

            # Stage: SPG semantic_prior
            spg_out = model.spg(q_emb, support_memory, dense_pe)
            intermediates["spg_semantic_prior"] = spg_out.semantic_prior

            # Stage: PromptFusion dense_prompt
            if model.prompt_fusion is not None and model.geometric_prior is not None:
                dp, st = model.prompt_fusion(
                    intermediates["geometric_prior"], spg_out.semantic_prior)
                intermediates["promptfusion_dense"] = dp
            else:
                dp = model._build_dense_prompt(
                    support_memory, sup_feat, sup_mask)
                if dp is None:
                    dp = spg_out.semantic_prior
                intermediates["promptfusion_dense"] = dp

            per_class[cls] = intermediates

        if len(per_class) < 2:
            continue

        # Pairwise comparison at each stage
        cls_ids = sorted(per_class.keys())
        stage_names = ["prototype", "support_memory", "geometric_prior",
                       "spg_semantic_prior", "promptfusion_dense"]

        for ci in range(len(cls_ids)):
            for cj in range(ci + 1, len(cls_ids)):
                ca, cb = cls_ids[ci], cls_ids[cj]
                name_a = ISAID5I_CATEGORIES.get(ca, str(ca))
                name_b = ISAID5I_CATEGORIES.get(cb, str(cb))

                for sname in stage_names:
                    ta = per_class[ca].get(sname)
                    tb = per_class[cb].get(sname)
                    if ta is None or tb is None:
                        continue

                    if sname in ("prototype", "support_memory"):
                        # [C] vectors → direct cosine
                        cos_val = float(F.cosine_similarity(
                            ta.unsqueeze(0), tb.unsqueeze(0)).item())
                    else:
                        # [1, C, H, W] tensors → per-channel mean cosine
                        cos_val = cosine_between_tensors(ta, tb)

                    stage_data[sname].append((cos_val, name_a, name_b))

    # ═══════════════════════════════════════════════════════════════
    # Per-tile detail (first 3)
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'─' * 72}")
    print("Per-Tile Detail (first 3 queries, first class pair each)")
    print("─" * 72)

    stage_display = ["prototype", "support_memory", "geometric_prior",
                     "spg_semantic_prior", "promptfusion_dense"]
    stage_labels = ["Prototype", "SupportMem", "GeoPrior", "SPG", "PromptFusion"]

    printed = 0
    for qi, (tile_idx, n_cls_on_tile, cls_list) in enumerate(selected[:3]):
        if printed >= 3:
            break
        sample = dataset[tile_idx]
        x, _ = preprocess_image(sample["image"])
        with torch.no_grad():
            q_emb = backbone(x.unsqueeze(0).to(device))["image_embedding"]
            if adapter is not None:
                q_emb = adapter(q_emb)
        dense_pe = model.sam_decoder.prompt_encoder.get_dense_pe()

        cls_present = cls_list[:4]
        if len(cls_present) < 2:
            continue

        ca, cb = cls_present[0], cls_present[1]
        name_a = ISAID5I_CATEGORIES.get(ca, str(ca))
        name_b = ISAID5I_CATEGORIES.get(cb, str(cb))

        per_class_local = {}
        for cls in [ca, cb]:
            sup_data = build_support_for_class(
                dataset, cls, args.k_shot, device, backbone, adapter, rng)
            if sup_data is None:
                break
            sup_feat, sup_mask = sup_data
            support_memory = model.support_encoder(sup_feat, sup_mask)
            intermediates = {}
            proto = compute_prototype(sup_feat, sup_mask)
            intermediates["prototype"] = proto
            intermediates["support_memory"] = support_memory.mean(dim=0)
            gp = model.geometric_prior(q_emb, support_memory) if model.geometric_prior else None
            intermediates["geometric_prior"] = gp
            spg_out = model.spg(q_emb, support_memory, dense_pe)
            intermediates["spg_semantic_prior"] = spg_out.semantic_prior
            if model.prompt_fusion is not None and gp is not None:
                dp, _ = model.prompt_fusion(gp, spg_out.semantic_prior)
            else:
                dp = model._build_dense_prompt(support_memory, sup_feat, sup_mask)
                if dp is None:
                    dp = spg_out.semantic_prior
            intermediates["promptfusion_dense"] = dp
            per_class_local[cls] = intermediates

        if len(per_class_local) < 2:
            continue

        print(f"\n  Query tile {tile_idx}: {name_a} vs {name_b}")
        for sname, slabel in zip(stage_display, stage_labels):
            ta = per_class_local[ca].get(sname)
            tb = per_class_local[cb].get(sname)
            if ta is None or tb is None:
                print(f"    {slabel:<15s}: N/A")
                continue
            if sname in ("prototype", "support_memory"):
                cos = float(F.cosine_similarity(
                    ta.unsqueeze(0), tb.unsqueeze(0)).item())
            else:
                cos = cosine_between_tensors(ta, tb)
            bar = "█" * int((1 - cos) * 50)
            print(f"    {slabel:<15s}: cos={cos:.4f}  {bar}")

        printed += 1

    # ═══════════════════════════════════════════════════════════════
    # Aggregate statistics
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 72}")
    print("PIPELINE FIDELITY SUMMARY")
    print("=" * 72)
    print(f"\n{'Stage':<20s}  {'Mean Cos':>9s}  {'Std Cos':>9s}  "
          f"{'Min Cos':>9s}  {'Max Cos':>9s}  {'Diagnosis':>30s}")
    print("-" * 90)

    for sname, slabel in zip(stage_display, stage_labels):
        vals = stage_data.get(sname, [])
        if not vals:
            continue
        coses = [v[0] for v in vals]
        mean_c = np.mean(coses)
        std_c = np.std(coses)
        min_c = np.min(coses)
        max_c = np.max(coses)

        if mean_c < 0.3:
            diag = "✅ STRONG class separation"
        elif mean_c < 0.7:
            diag = "⚠️  MODERATE separation"
        elif mean_c < 0.95:
            diag = "❌ WEAK — class being washed out"
        else:
            diag = "‼️  COLLAPSED — all classes identical"

        print(f"  {slabel:<18s}  {mean_c:>9.4f}  {std_c:>9.4f}  "
              f"{min_c:>9.4f}  {max_c:>9.4f}  {diag}")

    # ── Delta analysis: where does the biggest drop happen? ──
    print(f"\n{'─' * 72}")
    print("Step-wise Delta: cos_increase = this_stage - previous_stage")
    print("  (positive = class info lost in this step)")
    print("─" * 72)

    prev_mean = None
    prev_name = None
    for sname, slabel in zip(stage_display, stage_labels):
        vals = stage_data.get(sname, [])
        if not vals:
            continue
        coses = [v[0] for v in vals]
        mean_c = np.mean(coses)

        if prev_mean is not None:
            delta = mean_c - prev_mean
            if delta > 0.1:
                signal = f"⚠️  LOSES {(delta*100):.0f}% class info HERE"
            elif delta > 0.02:
                signal = f"~ minor loss ({(delta*100):.0f}%)"
            else:
                signal = "✓ preserves class info"
            print(f"  {prev_name} → {slabel}: Δcos={delta:+.4f}  {signal}")

        prev_mean = mean_c
        prev_name = slabel


if __name__ == "__main__":
    main()
