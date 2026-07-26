"""
Appearance vs Semantic Matching 诊断 | Appearance vs Semantic Matching Diagnostic.
==================================================================================

核心问题：模型学到的是"找和 support 相似的区域"还是"找 support 类别的像素"？

方法：对每个中间表示（GeoPrior / SPG / DensePrompt），计算其响应热图与：
  - 特定类别 GT（semantic）：只含 support 类别的 GT mask
  - 全前景 GT（objectness）：所有可见类别的 union

如果响应热图与 objectness 的 IoU >> 与 semantic 的 IoU：
  → 模型在做 appearance matching，不是 semantic matching
  → 这解释了 FB-IoU 高 (0.5) 但 mIoU 低 (0.1) 的根本原因

用法 | Usage:
    python tools/analysis/diag_appearance_vs_semantic.py \
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

def build_support_for_class(
    dataset, class_id: int, k_shot: int, device: torch.device,
    backbone: MobileSAMBackbone, adapter, rng: random.Random,
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
        [resize_mask(m, (feats.shape[2], feats.shape[3])).to(device) for m in masks], dim=0)
    if masks_grid.sum() < 1.0:
        return None
    return feats, masks_grid


def compute_prototype(features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Masked-mean prototype [C]."""
    masked = features * masks.unsqueeze(1)
    return masked.sum(dim=(0, 2, 3)) / (masks.sum() + 1e-8)


def tensor_to_response_map(
    features: torch.Tensor,    # [1, C, H, W]
    prototype: torch.Tensor,   # [C]
) -> np.ndarray:
    """Convert a dense feature map to a 2D response map via cosine with prototype.

    Returns: [H, W] float numpy array (cosine similarity per pixel).
    """
    C, H, W = features.shape[1:]
    flat = features.reshape(1, C, -1)              # [1, C, N]
    flat_n = F.normalize(flat, dim=1)              # [1, C, N]
    proto_n = F.normalize(prototype, dim=0).reshape(1, C, 1)  # [1, C, 1]
    sim = (flat_n * proto_n).sum(dim=1)             # [1, N]
    return sim.reshape(H, W).detach().cpu().numpy()  # [H, W]


def dice_coef(pred_map: np.ndarray, gt_mask: np.ndarray) -> float:
    """Dice coefficient between a continuous prediction map and binary GT."""
    # pred_map: [H, W] continuous, gt_mask: [H, W] bool
    pred_norm = (pred_map - pred_map.min()) / (pred_map.max() - pred_map.min() + 1e-8)
    pred_bin = (pred_norm > 0.5).astype(bool)
    inter = float((pred_bin & gt_mask).sum())
    return 2 * inter / max(float(pred_bin.sum() + gt_mask.sum()), 1.0)


def iou_coef(pred_map: np.ndarray, gt_mask: np.ndarray) -> float:
    """IoU between binarized prediction map and binary GT."""
    pred_norm = (pred_map - pred_map.min()) / (pred_map.max() - pred_map.min() + 1e-8)
    pred_bin = (pred_norm > 0.5).astype(bool)
    inter = float((pred_bin & gt_mask).sum())
    union = float((pred_bin | gt_mask).sum())
    return inter / max(union, 1.0)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Appearance vs Semantic Matching")
    parser.add_argument("--stage2-ckpt", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--n-queries", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    print(f"Device: {device}")

    # ── Load ──
    ckpt_path = Path(args.stage2_ckpt)
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

    adapter_state = ckpt.get("cat_adapter")
    if adapter_state is not None:
        adapter_cfg = ckpt.get("config", {}).get("adapter", {})
        adapter = CATAdapter(
            dim=256, bottleneck=int(adapter_cfg.get("bottleneck", 64))).to(device)
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

    # ── Find multi-class query tiles ──
    tile_scores = []
    for idx in range(len(dataset)):
        classes_present = []
        for cls in val_classes:
            gt = dataset.get_class_mask(idx, cls)
            if gt is not None and gt.sum() > 100:  # substantial FG
                classes_present.append(cls)
        if len(classes_present) >= 2:
            tile_scores.append((idx, len(classes_present), classes_present))

    tile_scores.sort(key=lambda x: -x[1])
    n_queries = min(args.n_queries, len(tile_scores))
    selected = tile_scores[:n_queries]
    print(f"Multi-class tiles: {len(tile_scores)}, using {n_queries}")

    # ═══════════════════════════════════════════════════════════════
    # Main experiment
    # ═══════════════════════════════════════════════════════════════

    # Per-stage: list of (IoU_semantic, IoU_objectness) tuples
    stage_iou_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    stage_dice_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)

    print(f"\n{'=' * 72}")
    print("Appearance vs Semantic Matching")
    print("  For each support class on a multi-class query tile:")
    print("  → Does the response map target the SPECIFIC GT class")
    print("    or ALL foreground indiscriminately?")
    print("=" * 72)

    for tile_idx, n_cls_on_tile, cls_list in tqdm(selected, desc="queries"):
        sample = dataset[tile_idx]
        x, _ = preprocess_image(sample["image"])

        # Build objectness GT (CPU-only, safe outside no_grad)
        gt_objectness = np.zeros((256, 256), dtype=bool)
        gt_per_class = {}
        for cls in val_classes:
            gt = dataset.get_class_mask(tile_idx, cls)
            if gt is not None:
                gt_np = gt.numpy().astype(bool)
                gt_per_class[cls] = gt_np
                gt_objectness = gt_objectness | gt_np

        if gt_objectness.sum() < 100:
            continue

        with torch.no_grad():
            q_emb = backbone(x.unsqueeze(0).to(device))["image_embedding"]
            if adapter is not None:
                q_emb = adapter(q_emb)

            dense_pe = model.sam_decoder.prompt_encoder.get_dense_pe()

            # For each class present on this tile
            for sup_cls in cls_list[:4]:
                if sup_cls not in gt_per_class:
                    continue
                gt_semantic = gt_per_class[sup_cls]  # specific class GT

                sup_data = build_support_for_class(
                    dataset, sup_cls, args.k_shot, device, backbone, adapter, rng)
                if sup_data is None:
                    continue
                sup_feat, sup_mask = sup_data

                # Prototype
                proto = compute_prototype(sup_feat, sup_mask)

                # SupportMemory
                support_memory = model.support_encoder(sup_feat, sup_mask)

                # GeoPrior
                gp = model.geometric_prior(q_emb, support_memory) if model.geometric_prior else None

                # SPG
                spg_out = model.spg(q_emb, support_memory, dense_pe)

                # PromptFusion
                if model.prompt_fusion is not None and gp is not None:
                    dp, _ = model.prompt_fusion(gp, spg_out.semantic_prior)
                else:
                    dp = model._build_dense_prompt(support_memory, sup_feat, sup_mask)
                    if dp is None:
                        dp = spg_out.semantic_prior

                # Decoder mask (final output)
                if model.bypass_head is not None:
                    low_res = model.bypass_head(dp)
                else:
                    low_res, _ = model.sam_decoder(q_emb, dp.mean(dim=(2, 3)), dp)
                final_mask = (low_res[0, 0] > 0).cpu().numpy()  # [256, 256] bool

                # Convert each intermediate to a response map
                proto_resp = tensor_to_response_map(q_emb, proto)
                stages = {}
                stages["prototype"] = resize_mask(proto_resp, (256, 256)).numpy()

                if gp is not None:
                    gp_resp = tensor_to_response_map(gp, proto)
                    stages["geometric_prior"] = resize_mask(gp_resp, (256, 256)).numpy()

                spg_resp = tensor_to_response_map(spg_out.semantic_prior, proto)
                stages["spg_semantic_prior"] = resize_mask(spg_resp, (256, 256)).numpy()

                dp_resp = tensor_to_response_map(dp, proto)
                stages["dense_prompt"] = resize_mask(dp_resp, (256, 256)).numpy()

                stages["final_mask"] = final_mask.astype(float)

                # For each stage, compute IoU against semantic GT and objectness GT
                for sname, resp_map in stages.items():
                    if sname == "final_mask":
                        iou_sem = float((resp_map.astype(bool) & gt_semantic).sum()) / \
                                  max(float((resp_map.astype(bool) | gt_semantic).sum()), 1.0)
                        iou_obj = float((resp_map.astype(bool) & gt_objectness).sum()) / \
                                  max(float((resp_map.astype(bool) | gt_objectness).sum()), 1.0)
                    else:
                        iou_sem = iou_coef(resp_map, gt_semantic)
                        iou_obj = iou_coef(resp_map, gt_objectness)

                    stage_iou_pairs[sname].append((iou_sem, iou_obj))

                # ── Free GPU tensors for this support class ──
                del sup_feat, sup_mask, proto, support_memory, spg_out, dp, low_res
                if gp is not None:
                    del gp
                    gp = None

        # ── Free per-tile GPU tensors ──
        del q_emb, dense_pe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════
    # Report
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 72}")
    print("RESULTS: Semantic vs Objectness Matching")
    print("=" * 72)

    stage_labels = {
        "prototype": "Prototype (baseline: what ideal semantic matching looks like)",
        "geometric_prior": "GeoPrior",
        "spg_semantic_prior": "SPG semantic_prior",
        "dense_prompt": "Dense Prompt (PromptFusion)",
        "final_mask": "Final Mask (decoder output)",
    }

    print(f"\n{'Stage':<20s}  {'IoU_semantic':>12s}  {'IoU_objectness':>14s}  "
          f"{'Ratio(sem/obj)':>14s}  {'Diagnosis':>30s}")
    print("-" * 100)

    for sname in ["prototype", "geometric_prior", "spg_semantic_prior",
                  "dense_prompt", "final_mask"]:
        pairs = stage_iou_pairs.get(sname, [])
        if not pairs:
            continue
        sem_vals = [p[0] for p in pairs]
        obj_vals = [p[1] for p in pairs]
        mean_sem = np.mean(sem_vals)
        mean_obj = np.mean(obj_vals)
        ratio = mean_sem / max(mean_obj, 1e-8)
        n = len(pairs)

        # Diagnosis
        if ratio > 0.8:
            diag = f"✅ SEMANTIC (n={n})"
        elif ratio > 0.5:
            diag = f"⚠️  MIXED (n={n})"
        elif ratio > 0.3:
            diag = f"❌ WEAKLY semantic (n={n})"
        else:
            diag = f"‼️  APPEARANCE-only (n={n})"

        print(f"  {sname:<18s}  {mean_sem:>12.4f}  {mean_obj:>14.4f}  "
              f"{ratio:>14.3f}  {diag}")

    # ── Gain/Loss relative to prototype ──
    print(f"\n{'─' * 72}")
    print("Information Gain/Loss relative to Prototype baseline:")
    proto_pairs = stage_iou_pairs.get("prototype", [])
    if proto_pairs:
        proto_sem = np.mean([p[0] for p in proto_pairs])
        for sname in ["geometric_prior", "spg_semantic_prior", "dense_prompt", "final_mask"]:
            pairs = stage_iou_pairs.get(sname, [])
            if not pairs:
                continue
            sem = np.mean([p[0] for p in pairs])
            delta = (sem - proto_sem) / max(proto_sem, 1e-8) * 100
            label = stage_labels.get(sname, sname)
            if delta < -50:
                print(f"  {label}: {delta:+.0f}% ↓↓↓  (most semantic info LOST)")
            elif delta < -20:
                print(f"  {label}: {delta:+.0f}% ↓↓")
            elif delta < -5:
                print(f"  {label}: {delta:+.0f}% ↓")
            else:
                print(f"  {label}: {delta:+.0f}%")

    # ── Semantic vs Objectness gap ──
    print(f"\n{'─' * 72}")
    print("Semantic — Objectness Gap (higher = better class specificity):")
    for sname in ["prototype", "geometric_prior", "spg_semantic_prior",
                  "dense_prompt", "final_mask"]:
        pairs = stage_iou_pairs.get(sname, [])
        if not pairs:
            continue
        sem = np.mean([p[0] for p in pairs])
        obj = np.mean([p[1] for p in pairs])
        gap = sem - obj
        label = stage_labels.get(sname, sname)
        bar = "█" * max(1, int(abs(gap) * 200))
        print(f"  {label:<18s}: gap={gap:+.4f}  {bar}")


if __name__ == "__main__":
    main()
