"""
Dense Prompt 类别敏感性诊断 | Dense Prompt Class Sensitivity Diagnostic.
========================================================================

三个实验，回答一个核心问题：Dense Prompt 编码的是"哪里是物体"还是"哪个类别"？

  Exp 1: 固定 Query，改变 Support 类别 → 比较 Dense Prompt（余弦相似度）
  Exp 2: 固定 Query，改变 Support 类别 → 比较最终 Mask（IoU）
  Exp 3: Dense Prompt 置零 → mIoU 变化（Decoder 是否真的依赖 Prompt）

用法 | Usage:
    python tools/analysis/diag_class_sensitivity.py \
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
    """Mean cosine similarity between two tensors of same shape, computed per-channel."""
    # a, b: [C, H, W] or [1, C, H, W]
    if a.ndim == 4:
        a = a[0]
        b = b[0]
    C = a.shape[0]
    a_flat = a.reshape(C, -1).float()
    b_flat = b.reshape(C, -1).float()
    a_n = F.normalize(a_flat, dim=1)
    b_n = F.normalize(b_flat, dim=1)
    return float((a_n * b_n).sum(dim=1).mean().item())


def mask_iou(pred: np.ndarray, other: np.ndarray) -> float:
    """IoU between two binary masks."""
    inter = float((pred & other).sum())
    union = float((pred | other).sum())
    return inter / max(union, 1.0)


def build_support_for_class(
    dataset, class_id: int, k_shot: int, device: torch.device,
    backbone: MobileSAMBackbone, adapter,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Build support features + masks for a class (FSS protocol)."""
    tiles = dataset.class_to_tiles(class_id)
    if len(tiles) < 1:
        return None

    # Scene-disjoint sampling
    scenes: dict[str, list[int]] = defaultdict(list)
    for idx in tiles:
        tile_id = dataset.tile_ids[idx]
        src = dataset._source_images.get(tile_id, str(idx))
        scenes[src].append(idx)

    chosen = []
    scene_list = list(scenes.keys())
    random.shuffle(scene_list)
    for src in scene_list:
        if len(chosen) >= k_shot:
            break
        chosen.append(random.choice(scenes[src]))

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
# Core diagnostic logic
# ═══════════════════════════════════════════════════════════════════

def compute_dense_prompt(
    model: AdaSAMModel,
    query_features: torch.Tensor,
    support_features: torch.Tensor,
    support_masks: torch.Tensor,
    use_raw_cosine: bool = False,
) -> dict:
    """Compute dense_prompt by calling model components directly.

    Returns dict with keys: dense_prompt, sparse_token, geometric_prior, semantic_prior
    """
    result = {}

    support_memory = model.support_encoder(support_features, support_masks)

    if use_raw_cosine:
        B, C, H, W = query_features.shape
        masked = support_features * support_masks.unsqueeze(1)
        proto = masked.sum(dim=(0, 2, 3)) / (support_masks.sum() + 1e-8)
        q_flat = query_features.reshape(B, C, -1)
        sim = torch.einsum("bcn,c->bn", F.normalize(q_flat, dim=1), F.normalize(proto, dim=0))
        s_min, s_max = sim.min(dim=1, keepdim=True)[0], sim.max(dim=1, keepdim=True)[0]
        sim = (sim - s_min) / (s_max - s_min + 1e-8)
        rsp_map = sim.reshape(B, 1, H, W)
        result["dense_prompt"] = rsp_map.expand(-1, C, -1, -1)
        result["sparse_token"] = result["dense_prompt"].mean(dim=(2, 3))
        result["geometric_prior"] = None
        result["semantic_prior"] = result["dense_prompt"]
        return result

    # Geometric Prior
    if model.geometric_prior is not None:
        gp = model.geometric_prior(query_features, support_memory)
        result["geometric_prior"] = gp
    else:
        gp = None
        result["geometric_prior"] = None

    # SPG
    dense_pe = model.sam_decoder.prompt_encoder.get_dense_pe()
    spg_out = model.spg(query_features, support_memory, dense_pe)
    result["semantic_prior"] = spg_out.semantic_prior

    # PromptFusion
    if model.prompt_fusion is not None and gp is not None:
        dense_prompt, sparse_token = model.prompt_fusion(gp, spg_out.semantic_prior)
    else:
        dense_prompt = model._build_dense_prompt(support_memory, support_features, support_masks)
        if dense_prompt is None:
            dense_prompt = spg_out.semantic_prior
        sparse_token = dense_prompt.mean(dim=(2, 3))

    result["dense_prompt"] = dense_prompt
    result["sparse_token"] = sparse_token
    return result


def decode_mask(
    model: AdaSAMModel,
    query_features: torch.Tensor,
    dense_prompt: torch.Tensor,
    sparse_token: torch.Tensor,
) -> torch.Tensor:
    """Decode mask from dense_prompt using either bypass_head or SAM decoder."""
    if model.bypass_head is not None:
        low_res = model.bypass_head(dense_prompt)  # [1, 1, 256, 256]
    else:
        low_res, _ = model.sam_decoder(query_features, sparse_token, dense_prompt)
    # Return binarized mask at 256x256
    mask = (low_res[0, 0] > 0).cpu().numpy()  # [256, 256] bool
    return mask


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Dense Prompt Class Sensitivity Diagnostic"
    )
    parser.add_argument("--stage2-ckpt", required=True,
                        help="Path to Stage 2 checkpoint")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--n-queries", type=int, default=5,
                        help="Number of query tiles to test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-cosine", action="store_true",
                        help="Use raw cosine prompt instead of SPG+GeoPrior")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load checkpoint ──
    ckpt_path = Path(args.stage2_ckpt)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    abl = cfg.get("ablation", {})
    print(f"  epoch: {ckpt.get('epoch', '?')}")
    print(f"  fold: {ckpt.get('fold', '?')}, k_shot: {ckpt.get('k_shot', '?')}")
    print(f"  raw_cosine: {abl.get('raw_cosine', False)}")
    print(f"  bypass_decoder: {abl.get('bypass_decoder', False)}")

    # ── Build model ──
    weights_path = str(_REPO_ROOT / cfg.get("backbone", {}).get("checkpoint", "weights/mobile_sam.pt"))
    sam = build_mobile_sam(weights_path, cfg.get("backbone", {}).get("model_type", "vit_t"), device)
    backbone = MobileSAMBackbone(sam.image_encoder, sam.image_encoder.img_size).to(device)

    model_cfg = AdaSAMModelConfig.from_dict(cfg)
    model = AdaSAMModel(sam, model_cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"  bypass_head active: {model.bypass_head is not None}")

    # ── Load adapter ──
    adapter_state = ckpt.get("cat_adapter")
    if adapter_state is not None:
        adapter_cfg = ckpt.get("config", {}).get("adapter", {})
        adapter = CATAdapter(dim=256, bottleneck=int(adapter_cfg.get("bottleneck", 64))).to(device)
        adapter.load_state_dict(adapter_state)
        adapter.eval()
        for p in adapter.parameters():
            p.requires_grad_(False)
        print("  adapter: loaded + frozen")
    else:
        adapter = None
        print("  adapter: NONE (raw SAM features)")

    # ── Dataset ──
    data_root = str(_REPO_ROOT / args.data_root) if not Path(args.data_root).is_absolute() else args.data_root
    dataset = ISAID5iDataset(root=data_root, fold=args.fold, split="val", mode="base")
    val_classes = dataset.visible_classes()
    print(f"Val classes: {val_classes}")
    for cls in val_classes:
        name = ISAID5I_CATEGORIES.get(cls, f"cls{cls}")
        n_tiles = len(dataset.class_to_tiles(cls))
        print(f"  class {cls:>2d} ({name:<20s}): {n_tiles} tiles")

    # ── Find query tiles with multiple GT classes ──
    # Score each tile by how many val_classes have GT on it
    tile_scores = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        classes_present = []
        for cls in val_classes:
            gt = dataset.get_class_mask(idx, cls)
            if gt is not None and gt.sum() > 50:  # at least 50 FG pixels
                classes_present.append(cls)
        if len(classes_present) >= 2:
            tile_scores.append((idx, len(classes_present), classes_present))

    tile_scores.sort(key=lambda x: -x[1])
    print(f"\nTiles with ≥2 classes (FG>50px): {len(tile_scores)}")
    for idx, n_cls, cls_list in tile_scores[:10]:
        names = [ISAID5I_CATEGORIES.get(c, str(c)) for c in cls_list]
        print(f"  tile {idx}: {n_cls} classes → {names}")

    if len(tile_scores) < args.n_queries:
        print(f"WARNING: only {len(tile_scores)} multi-class tiles available")
    n_queries = min(args.n_queries, len(tile_scores))
    selected = tile_scores[:n_queries]

    # ═══════════════════════════════════════════════════════════════
    # EXP 1+2: Cross-Class Sensitivity
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 72)
    print("EXP 1+2: Cross-Class Dense Prompt & Mask Sensitivity")
    print("=" * 72)

    all_prompt_cos = []   # per-tile, per-class-pair: prompt cosine
    all_mask_iou = []     # per-tile, per-class-pair: mask IoU

    for tile_idx, n_cls, cls_list in selected:
        sample = dataset[tile_idx]

        # Embed query once
        x, _ = preprocess_image(sample["image"])
        with torch.no_grad():
            q_emb = backbone(x.unsqueeze(0).to(device))["image_embedding"]
            if adapter is not None:
                q_emb = adapter(q_emb)

        # Build support and compute dense_prompt + mask for each class
        per_class = {}  # cls → {"dense_prompt": ..., "mask": ...}
        for cls in cls_list:
            sup_data = build_support_for_class(
                dataset, cls, args.k_shot, device, backbone, adapter
            )
            if sup_data is None:
                continue
            sup_feat, sup_mask = sup_data

            result = compute_dense_prompt(
                model, q_emb, sup_feat, sup_mask, use_raw_cosine=args.raw_cosine
            )
            dp = result["dense_prompt"]
            st = result["sparse_token"]
            mask = decode_mask(model, q_emb, dp, st)

            per_class[cls] = {
                "dense_prompt": dp,
                "mask": mask,
            }

        if len(per_class) < 2:
            continue

        # Pairwise comparison
        cls_ids = sorted(per_class.keys())
        tile_cos = []
        tile_iou = []
        for i in range(len(cls_ids)):
            for j in range(i + 1, len(cls_ids)):
                ca, cb = cls_ids[i], cls_ids[j]
                cos_sim = cosine_between_tensors(
                    per_class[ca]["dense_prompt"],
                    per_class[cb]["dense_prompt"],
                )
                miou = mask_iou(per_class[ca]["mask"], per_class[cb]["mask"])
                tile_cos.append(cos_sim)
                tile_iou.append(miou)
                name_a = ISAID5I_CATEGORIES.get(ca, str(ca))
                name_b = ISAID5I_CATEGORIES.get(cb, str(cb))
                verdict = "⚠️ VERY SIMILAR" if cos_sim > 0.95 else ("✓ DIFFERENT" if cos_sim < 0.5 else "~ MODERATE")
                print(f"  tile={tile_idx}  {name_a:>20s} vs {name_b:<20s}  "
                      f"prompt_cos={cos_sim:.4f} ({verdict})  mask_IoU={miou:.4f}")

        all_prompt_cos.extend(tile_cos)
        all_mask_iou.extend(tile_iou)

    # ── Summary ──
    print(f"\n{'─' * 60}")
    print(f"EXP 1: Dense Prompt cross-class cosine (lower = more class-specific)")
    print(f"  mean={np.mean(all_prompt_cos):.4f}  std={np.std(all_prompt_cos):.4f}  "
          f"min={np.min(all_prompt_cos):.4f}  max={np.max(all_prompt_cos):.4f}")
    if np.mean(all_prompt_cos) > 0.95:
        print(f"  ❌ Dense Prompt is CLASS-AGNOSTIC (cos > 0.95)")
    elif np.mean(all_prompt_cos) > 0.7:
        print(f"  ⚠️  Dense Prompt is WEAKLY class-specific (0.7 < cos < 0.95)")
    else:
        print(f"  ✅ Dense Prompt IS class-specific (cos < 0.7)")

    print(f"\nEXP 2: Mask cross-class IoU (lower = support class matters more)")
    print(f"  mean={np.mean(all_mask_iou):.4f}  std={np.std(all_mask_iou):.4f}  "
          f"min={np.min(all_mask_iou):.4f}  max={np.max(all_mask_iou):.4f}")
    if np.mean(all_mask_iou) > 0.8:
        print(f"  ❌ Support class has NO EFFECT on predicted masks (IoU > 0.8)")
    elif np.mean(all_mask_iou) > 0.5:
        print(f"  ⚠️  Support class has WEAK effect on masks (0.5 < IoU < 0.8)")
    else:
        print(f"  ✅ Support class STRONGLY changes masks (IoU < 0.5)")

    # ═══════════════════════════════════════════════════════════════
    # EXP 3: Zero Dense Prompt Ablation
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 72}")
    print("EXP 3: Zero Dense Prompt Ablation")
    print("=" * 72)

    ious_normal = []
    ious_zero = []

    for tile_idx, n_cls, cls_list in selected[:3]:  # use first 3 tiles
        sample = dataset[tile_idx]
        x, _ = preprocess_image(sample["image"])
        with torch.no_grad():
            q_emb = backbone(x.unsqueeze(0).to(device))["image_embedding"]
            if adapter is not None:
                q_emb = adapter(q_emb)

        for cls in cls_list:
            gt = dataset.get_class_mask(tile_idx, cls)
            if gt is None or gt.sum() < 50:
                continue
            gt_np = gt.numpy().astype(bool)

            sup_data = build_support_for_class(
                dataset, cls, args.k_shot, device, backbone, adapter
            )
            if sup_data is None:
                continue
            sup_feat, sup_mask = sup_data

            result = compute_dense_prompt(
                model, q_emb, sup_feat, sup_mask, use_raw_cosine=args.raw_cosine
            )
            dp = result["dense_prompt"]
            st = result["sparse_token"]

            # Normal
            mask_normal = decode_mask(model, q_emb, dp, st)
            iou_n = mask_iou(mask_normal, gt_np)

            # Zeroed
            dp_zero = torch.zeros_like(dp)
            mask_zero = decode_mask(model, q_emb, dp_zero, st)
            iou_z = mask_iou(mask_zero, gt_np)

            if iou_n > 0.01:  # only count if normal has some signal
                ious_normal.append(iou_n)
                ious_zero.append(iou_z)

            name = ISAID5I_CATEGORIES.get(cls, str(cls))
            print(f"  tile={tile_idx} cls={name:<20s}  "
                  f"IoU_normal={iou_n:.4f}  IoU_zero={iou_z:.4f}  "
                  f"drop={(iou_n - iou_z) / max(iou_n, 1e-4):.1%}")

    print(f"\n{'─' * 60}")
    if ious_normal:
        mean_n = np.mean(ious_normal)
        mean_z = np.mean(ious_zero)
        drop = (mean_n - mean_z) / max(mean_n, 1e-4)
        print(f"  mean IoU_normal={mean_n:.4f}  mean IoU_zero={mean_z:.4f}")
        print(f"  relative drop: {drop:.1%}")
        if drop < 0.1:
            print(f"  ❌ Dense Prompt is IGNORED by decoder (drop < 10%)")
        elif drop < 0.5:
            print(f"  ⚠️  Dense Prompt contributes partially ({drop:.0%} drop)")
        else:
            print(f"  ✅ Dense Prompt is ESSENTIAL for mask prediction ({drop:.0%} drop)")
    else:
        print("  No valid samples for ablation")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("DIAGNOSTIC SUMMARY")
    print("=" * 72)
    print(f"  Exp 1 - Prompt cross-class cos:  {np.mean(all_prompt_cos):.4f} "
          f"({'AGNOSTIC' if np.mean(all_prompt_cos) > 0.95 else 'SPECIFIC'})")
    print(f"  Exp 2 - Mask cross-class IoU:    {np.mean(all_mask_iou):.4f} "
          f"({'NO EFFECT' if np.mean(all_mask_iou) > 0.8 else 'AFFECTED'})")
    if ious_normal:
        print(f"  Exp 3 - Zero prompt IoU drop:   {drop:.1%} "
              f"({'IGNORED' if drop < 0.1 else 'ESSENTIAL' if drop > 0.5 else 'PARTIAL'})")


if __name__ == "__main__":
    main()
