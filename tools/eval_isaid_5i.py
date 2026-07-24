"""
iSAID-5i 小样本语义分割评估 | Few-shot Semantic Segmentation Evaluation.
=========================================================================

基于 AdaSAM Stage 2 架构在 iSAID-5i 标准小样本协议上评估, 遵循 FSS Benchmark
标准评估协议 (固定 Support、FB-IoU、3-fold 交叉验证、多 Seed Mean±Std)。

Evaluate AdaSAM Stage 2 architecture on the standard iSAID-5i few-shot protocol,
following the FSS Benchmark standard (fixed support, FB-IoU, 3-fold CV,
multi-seed Mean±Std).

用法 | Usage::

    # 单 fold 评估
    python tools/eval_isaid_5i.py --checkpoint <ckpt> --k-shot 5

    # 三折交叉验证 (Fold0/1/2/Mean)
    python tools/eval_isaid_5i.py --checkpoint <ckpt> --k-shot 5 --all-folds

    # 多 seed Mean±Std
    python tools/eval_isaid_5i.py --checkpoint <ckpt> --k-shot 5 --seeds 42 123 456

    # 全量: 3 folds × 3 seeds
    python tools/eval_isaid_5i.py --checkpoint <ckpt> --k-shot 5 --all-folds --seeds 42 123 456

    # 保存预测 + 诊断
    python tools/eval_isaid_5i.py --checkpoint <ckpt> --k-shot 5 --save-predictions --diagnostics

指标 | Metrics:
    - mIoU: 所有可见类的平均 IoU (核心指标)
    - FB-IoU: 前景-背景 IoU (FSS 标准指标)
    - Per-class IoU (含 GT# 和 Support#)
    - Fold0 / Fold1 / Fold2 / Mean (--all-folds)
    - Mean±Std (--seeds)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adasam.adapters import CATAdapter
from adasam.backbone import build_mobile_sam, MobileSAMBackbone
from adasam.datasets.isaid_5i import (
    ISAID5iDataset,
    ISAID5I_CATEGORIES,
    ISAID5I_FOLDS,
)
from adasam.model import AdaSAMModel, AdaSAMModelConfig
from adasam.utils import set_seed
from adasam.utils.transforms import preprocess_image, resize_mask


# ═══════════════════════════════════════════════════════════════════
# Support Cache — FSS 标准协议: 评估期间 Support 固定
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def build_support_cache(
    *,
    data_root: Path,
    fold: int,
    mode: str,
    k_shot: int,
    visible_classes: list[int],
    backbone: MobileSAMBackbone,
    cat_adapter: CATAdapter | None,
    support_seed: int,
    device: torch.device,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """构建固定 Support 缓存 | Build fixed support cache.

    FSS 标准协议: 每个类的 Support 在评估前一次性采样并编码, 整个评估期间保持不变。
    Standard FSS protocol: support samples are drawn once per class and held
    fixed for the entire evaluation — no per-query resampling.

    :return: {class_id: (support_features [K,C,gh,gw], support_masks_grid [K,gh,gw])}
    """
    train_ds = ISAID5iDataset(root=str(data_root), fold=fold, split="train", mode=mode)
    rng = random.Random(support_seed)

    cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for cls in visible_classes:
        tile_indices = train_ds.class_to_tiles(cls)
        if not tile_indices:
            continue

        # Group by source image for scene-disjoint sampling
        scenes: dict[str, list[int]] = defaultdict(list)
        for idx in tile_indices:
            src = train_ds._source_images.get(train_ds.tile_ids[idx],
                                               train_ds.tile_ids[idx])
            scenes[src].append(idx)

        # Sample K different scenes
        scene_keys = list(scenes)
        k = min(k_shot, len(scene_keys))
        chosen_scenes = rng.sample(scene_keys, k)

        images, masks = [], []
        for sid in chosen_scenes:
            idx = rng.choice(scenes[sid])
            sample = train_ds[idx]
            fg = _class_mask(train_ds, idx, cls)
            if fg is None or fg.sum() < 1:
                continue
            x, _ = preprocess_image(sample["image"])
            images.append(x.to(device))
            masks.append(fg)

        if not images:
            continue

        # Backbone
        feats = backbone(torch.stack(images, dim=0))["image_embedding"]  # [K,256,64,64]
        if cat_adapter is not None:
            feats = cat_adapter(feats)

        # Resize masks to feature grid
        masks_grid = torch.stack(
            [resize_mask(m, (feats.shape[2], feats.shape[3])).to(device) for m in masks],
            dim=0,
        )

        cache[cls] = (feats, masks_grid)

    return cache


def _class_mask(dataset: ISAID5iDataset, index: int, class_id: int) -> torch.Tensor | None:
    """Get merged binary mask of a given class from a tile (semantic: class-level merge)."""
    return dataset.get_class_mask(index, class_id)


# ═══════════════════════════════════════════════════════════════════
# FB-IoU: Foreground-Background IoU (FSS 标准指标)
# ═══════════════════════════════════════════════════════════════════

def compute_fb_iou(
    per_class_inter: dict[int, float],
    per_class_union: dict[int, float],
    visible_classes: list[int],
) -> dict[str, float]:
    """计算 FB-IoU | Compute Foreground-Background IoU.

    将所有可见 FG 类合并为 "前景", 与背景之间计算 IoU。
    Merge all visible FG classes into "foreground", compute IoU against background.

    FB-IoU = (FG-IoU + BG-IoU) / 2

    :param per_class_inter: {cls: intersection_sum} from accumulation.
    :param per_class_union: {cls: union_sum} from accumulation.
    :param visible_classes: list of class IDs considered "foreground".
    :return: {"FB-IoU": float, "FG-IoU": float, "BG-IoU": float}
    """
    fg_inter = sum(per_class_inter.get(c, 0.0) for c in visible_classes)
    fg_union = sum(per_class_union.get(c, 0.0) for c in visible_classes)

    bg_inter = per_class_inter.get(0, 0.0)
    bg_union = per_class_union.get(0, 0.0)

    fg_iou = fg_inter / fg_union if fg_union > 0 else float("nan")
    bg_iou = bg_inter / bg_union if bg_union > 0 else float("nan")

    valid = [v for v in [fg_iou, bg_iou] if v == v]
    fb_iou = float(np.mean(valid)) if valid else float("nan")

    return {"FB-IoU": round(fb_iou, 6), "FG-IoU": round(fg_iou, 6),
            "BG-IoU": round(bg_iou, 6)}


def compute_fb_from_accum(fg_inter: float, fg_union: float,
                          bg_inter: float, bg_union: float) -> dict[str, float]:
    """从已累积的 FG/BG inter/union 计算 FB-IoU | Compute FB-IoU from accumulated values."""
    fg_iou = fg_inter / fg_union if fg_union > 0 else float("nan")
    bg_iou = bg_inter / bg_union if bg_union > 0 else float("nan")
    valid = [v for v in [fg_iou, bg_iou] if v == v]
    fb_iou = float(np.mean(valid)) if valid else float("nan")
    return {"FB-IoU": round(fb_iou, 6), "FG-IoU": round(fg_iou, 6),
            "BG-IoU": round(bg_iou, 6)}


@torch.no_grad()
def _compute_split_fb_delta(
    pred_masks: dict[int, torch.Tensor],
    gt_masks: dict[int, torch.Tensor],
    class_ids: set[int],
    H: int,
    W: int,
    device: torch.device,
) -> dict[str, float]:
    """Compute FB-IoU deltas for a subset of classes (novel or base).

    Returns accumulated inter/union deltas for FG and BG on this tile.
    """
    # FG = union of only these classes
    fg_pred = torch.zeros(H, W, dtype=torch.bool, device=device)
    fg_gt = torch.zeros(H, W, dtype=torch.bool)
    for cls in class_ids:
        if cls in pred_masks:
            fg_pred = fg_pred | pred_masks[cls]
        if cls in gt_masks:
            fg_gt = fg_gt | gt_masks[cls].to(device)

    bg_pred = ~fg_pred
    bg_gt = ~fg_gt

    return {
        "fg_inter": float((fg_pred & fg_gt).sum().item()),
        "fg_union": float((fg_pred | fg_gt).sum().item()),
        "bg_inter": float((bg_pred & bg_gt).sum().item()),
        "bg_union": float((bg_pred | bg_gt).sum().item()),
    }


# ═══════════════════════════════════════════════════════════════════
# 单 Fold 评估 | Single Fold Evaluation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_one_fold(
    *,
    checkpoint_path: Path,
    data_root: Path,
    fold: int,
    mode: str,
    k_shot: int,
    seed: int,
    score_thr: float,
    device: torch.device,
    max_samples: int = 0,
) -> dict:
    """运行单个 (fold, seed) 的完整评估 | Run a complete single (fold, seed) evaluation.

    :param max_samples: 限制评估 tile 数 (0=全部), 用于冒烟测试 | limit tiles (0=all).
    :return: metrics dict with mIoU, FB-IoU, per-class IoU, etc.
    """
    set_seed(seed)

    # ── Load checkpoint ──
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    # ── Build model ──
    bb_cfg = cfg.get("backbone", {})
    bb_path = Path(bb_cfg.get("checkpoint", "weights/mobile_sam.pt"))
    bb_path = bb_path if bb_path.is_absolute() else _REPO_ROOT / bb_path

    sam = build_mobile_sam(str(bb_path), bb_cfg.get("model_type", "vit_t"), device)
    backbone = MobileSAMBackbone(sam.image_encoder, sam.image_encoder.img_size).to(device)
    embed_dim = int(cfg.get("support_encoder", {}).get("embed_dim", 256))
    model = AdaSAMModel(sam, AdaSAMModelConfig.from_dict(cfg)).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[eval] WARNING: missing keys (new code, old ckpt): {missing}")
    if unexpected:
        print(f"[eval] WARNING: unexpected keys (new ckpt, old code): {unexpected}")
    model.eval()

    # CAT-Adapter (optional)
    cat_adapter = None
    if ckpt.get("cat_adapter") is not None:
        tcfg = cfg.get("train", {})
        adapter_cfg = tcfg.get("cat_adapter", {})
        cat_adapter = CATAdapter(
            dim=embed_dim,
            bottleneck=int(adapter_cfg.get("bottleneck", 64)),
        ).to(device)
        cat_adapter.load_state_dict(ckpt["cat_adapter"])
        cat_adapter.eval()

    # ── Data ──
    val_ds = ISAID5iDataset(root=str(data_root), fold=fold, split="val", mode=mode)
    visible_classes = val_ds.visible_classes()

    # Determine novel/base split for mode="all"
    fold_def = ISAID5I_FOLDS[fold]
    novel_class_ids = set(fold_def["test"])
    base_class_ids = set(fold_def["train"])
    has_novel_base_split = (mode == "all")

    # ── Build support cache (FIXED per class for this evaluation run) ──
    support_cache = build_support_cache(
        data_root=data_root, fold=fold, mode=mode, k_shot=k_shot,
        visible_classes=visible_classes, backbone=backbone,
        cat_adapter=cat_adapter, support_seed=seed, device=device,
    )

    # Track support counts for output
    support_counts = {cls: cache[0].shape[0] for cls, cache in support_cache.items()}

    # ── Evaluate ──
    inter = {c: 0.0 for c in range(16)}   # per-class intersection
    union = {c: 0.0 for c in range(16)}   # per-class union
    gt_counts = {c: 0 for c in range(16)}  # number of tiles where class c appears
    correct_pixels = 0
    total_pixels = 0
    per_sample: list[dict] = []

    # Novel/Base FB-IoU accumulators (for mode="all")
    novel_fg_inter = 0.0; novel_fg_union = 0.0
    novel_bg_inter = 0.0; novel_bg_union = 0.0
    base_fg_inter = 0.0; base_fg_union = 0.0
    base_bg_inter = 0.0; base_bg_union = 0.0

    # Predictions for COCO-style JSON export
    all_predictions: list[dict] = []

    n_eval = min(len(val_ds), max_samples) if max_samples > 0 else len(val_ds)
    for idx in tqdm(range(n_eval), desc=f"fold{fold}"):
        sample = val_ds[idx]
        query_image = sample["image"]
        H, W = query_image.shape[1], query_image.shape[2]
        tile_id = sample.get("tile_id", str(idx))

        # Embed query
        x, meta = preprocess_image(query_image)
        query_emb = backbone(x.unsqueeze(0).to(device))["image_embedding"]
        if cat_adapter is not None:
            query_emb = cat_adapter(query_emb)

        # Build per-class GT masks (semantic: class-level merge)
        gt_masks: dict[int, torch.Tensor] = {}
        for cls in visible_classes:
            gt_m = val_ds.get_class_mask(idx, cls)
            if gt_m is not None:
                gt_masks[cls] = gt_m.to(device)

        # Predict each visible class
        pred_masks: dict[int, torch.Tensor] = {}
        sample_ious: dict[str, float | None] = {}

        for cls in visible_classes:
            cached = support_cache.get(cls)
            if cached is None:
                sample_ious[ISAID5I_CATEGORIES.get(cls, f"cls{cls}")] = None
                continue

            support_features, support_masks_grid = cached

            masks, scores = model.predict(
                query_features=query_emb,
                support_features=support_features,
                support_masks=support_masks_grid,
                input_size=meta.input_size,
                original_size=(H, W),
                score_thr=score_thr,
            )

            # SPG unified output: single mask [1, H, W] (no per-query aggregation needed)
            pred = masks[0] if masks.shape[0] > 0 else torch.zeros(H, W, dtype=torch.bool, device=device)
            pred_masks[cls] = pred

            # Accumulate per-class IoU
            gt = gt_masks.get(cls)
            if gt is not None:
                gt_t = gt.to(device)
                inter[cls] += (pred & gt_t).sum().item()
                union[cls] += (pred | gt_t).sum().item()
                gt_counts[cls] += 1
                iou = inter[cls] / union[cls] if union[cls] > 0 else float("nan")
            else:
                # Class not in this tile → predicted mask counts as FP
                union[cls] += pred.sum().item()
                iou = float("nan")

            sample_ious[ISAID5I_CATEGORIES.get(cls, f"cls{cls}")] = (
                round(iou, 6) if iou == iou else None
            )

        # Save predictions for COCO JSON export
        for cls, pred_m in pred_masks.items():
            if pred_m.sum() > 0:
                all_predictions.append({
                    "tile_id": tile_id,
                    "category_id": cls,
                    "category_name": ISAID5I_CATEGORIES.get(cls, f"cls{cls}"),
                    "mask_area": int(pred_m.sum().item()),
                    "score": 0.0,  # scores are per-support-set, not per-class here
                })

        # Novel/Base FB-IoU accumulation (for mode="all")
        if has_novel_base_split:
            d = _compute_split_fb_delta(pred_masks, gt_masks, novel_class_ids, H, W, device)
            novel_fg_inter += d["fg_inter"]; novel_fg_union += d["fg_union"]
            novel_bg_inter += d["bg_inter"]; novel_bg_union += d["bg_union"]
            d = _compute_split_fb_delta(pred_masks, gt_masks, base_class_ids, H, W, device)
            base_fg_inter += d["fg_inter"]; base_fg_union += d["fg_union"]
            base_bg_inter += d["bg_inter"]; base_bg_union += d["bg_union"]

        # Track BG (pixels not belonging to any visible class)
        gt_bg = torch.ones(H, W, dtype=torch.bool)
        for cls, m in gt_masks.items():
            if cls in visible_classes:
                gt_bg = gt_bg & ~m
        gt_bg_t = gt_bg.to(device)

        pred_bg = torch.ones(H, W, dtype=torch.bool, device=device)
        for cls, m in pred_masks.items():
            if cls in visible_classes:
                pred_bg = pred_bg & ~m

        inter[0] += (pred_bg & gt_bg_t).sum().item()
        union[0] += (pred_bg | gt_bg_t).sum().item()

        # Pixel accuracy (for debug log only)
        gt_combined = torch.zeros(H, W, dtype=torch.long)
        for cls, m in gt_masks.items():
            gt_combined[m] = cls
        pred_combined = torch.zeros(H, W, dtype=torch.long, device=device)
        for cls, m in pred_masks.items():
            pred_combined[m] = cls
        correct_pixels += (pred_combined == gt_combined.to(device)).sum().item()
        total_pixels += H * W

        # Per-sample mIoU (only classes present in GT)
        valid_ious = [v for v in sample_ious.values() if v is not None]
        per_sample.append({
            "tile_id": tile_id,
            "mIoU": round(float(np.mean(valid_ious)), 6) if valid_ious else 0.0,
            "per_class": sample_ious,
        })

    # ── Compute aggregate metrics ──
    per_class_iou = {}
    valid_ious_list = []
    novel_ious_list = []
    base_ious_list = []
    for cls in range(1, 16):
        u = union.get(cls, 0.0)
        if u > 0 and gt_counts.get(cls, 0) > 0:
            iou_c = inter[cls] / u
            per_class_iou[cls] = {
                "IoU": round(iou_c, 6),
                "GT_tiles": gt_counts[cls],
                "Support_tiles": support_counts.get(cls, 0),
            }
            valid_ious_list.append(iou_c)
            if has_novel_base_split:
                if cls in novel_class_ids:
                    novel_ious_list.append(iou_c)
                elif cls in base_class_ids:
                    base_ious_list.append(iou_c)
        elif cls in visible_classes:
            per_class_iou[cls] = {
                "IoU": None,
                "GT_tiles": gt_counts.get(cls, 0),
                "Support_tiles": support_counts.get(cls, 0),
            }

    miou = round(float(np.mean(valid_ious_list)), 6) if valid_ious_list else 0.0
    fb = compute_fb_iou(inter, union, visible_classes)

    # Novel/Base mIoU and FB-IoU
    novel_miou = round(float(np.mean(novel_ious_list)), 6) if novel_ious_list else None
    base_miou = round(float(np.mean(base_ious_list)), 6) if base_ious_list else None
    novel_fb = compute_fb_from_accum(novel_fg_inter, novel_fg_union,
                                     novel_bg_inter, novel_bg_union) if has_novel_base_split else None
    base_fb = compute_fb_from_accum(base_fg_inter, base_fg_union,
                                    base_bg_inter, base_bg_union) if has_novel_base_split else None

    sample_mious = [s["mIoU"] for s in per_sample]
    pa = round(correct_pixels / total_pixels, 6) if total_pixels > 0 else 0.0

    # Count GT tiles per class
    gt_tile_counts = {ISAID5I_CATEGORIES.get(c, f"cls{c}"): gt_counts.get(c, 0)
                      for c in visible_classes}

    return {
        "mIoU": miou,
        "FB-IoU": fb["FB-IoU"],
        "FG-IoU": fb["FG-IoU"],
        "BG-IoU": fb["BG-IoU"],
        "pixel_accuracy": pa,  # debug only
        "per_class_IoU": per_class_iou,
        "n_tiles": len(per_sample),
        "n_classes": len(visible_classes),
        "sample_mIoU_mean": round(float(np.mean(sample_mious)), 6) if sample_mious else 0.0,
        "sample_mIoU_median": round(float(np.median(sample_mious)), 6) if sample_mious else 0.0,
        "per_sample": per_sample,
        "predictions": all_predictions,
        # Novel/Base split (only when mode="all")
        "novel_mIoU": novel_miou,
        "base_mIoU": base_miou,
        "novel_FB-IoU": novel_fb["FB-IoU"] if novel_fb else None,
        "base_FB-IoU": base_fb["FB-IoU"] if base_fb else None,
        "support_cache_info": {
            cls: {"n_support": support_counts.get(cls, 0),
                  "n_gt_tiles": gt_counts.get(cls, 0)}
            for cls in visible_classes
        },
    }


# ═══════════════════════════════════════════════════════════════════
# Multi-Fold / Multi-Seed 编排 | Orchestration
# ═══════════════════════════════════════════════════════════════════

def run_folds(
    checkpoint_path: Path,
    data_root: Path,
    folds: list[int],
    mode: str,
    k_shot: int,
    seed: int,
    score_thr: float,
    device: torch.device,
    max_samples: int = 0,
) -> dict:
    """运行多个 fold 并汇总 | Run multiple folds and aggregate.

    :return: {"fold0": {...}, "fold1": {...}, "fold2": {...},
              "mean_mIoU": float, "mean_FB-IoU": float}
    """
    fold_results = {}
    for fold in folds:
        fold_results[f"fold{fold}"] = evaluate_one_fold(
            checkpoint_path=checkpoint_path, data_root=data_root,
            fold=fold, mode=mode, k_shot=k_shot, seed=seed,
            score_thr=score_thr, device=device, max_samples=max_samples,
        )

    mious = [r["mIoU"] for r in fold_results.values()]
    fbs = [r["FB-IoU"] for r in fold_results.values()]

    return {
        **fold_results,
        "mean_mIoU": round(float(np.mean(mious)), 6),
        "mean_FB-IoU": round(float(np.mean(fbs)), 6),
    }


def run_seeds(
    checkpoint_path: Path,
    data_root: Path,
    fold: int,
    mode: str,
    k_shot: int,
    seeds: list[int],
    score_thr: float,
    device: torch.device,
    max_samples: int = 0,
) -> dict:
    """运行多个 seed 并计算 Mean±Std | Run multiple seeds, compute Mean±Std.

    :return: {"per_seed": [...], "mean_mIoU": float, "std_mIoU": float, ...}
    """
    results = []
    for seed in seeds:
        r = evaluate_one_fold(
            checkpoint_path=checkpoint_path, data_root=data_root,
            fold=fold, mode=mode, k_shot=k_shot, seed=seed,
            score_thr=score_thr, device=device, max_samples=max_samples,
        )
        results.append(r)

    mious = [r["mIoU"] for r in results]
    fbs = [r["FB-IoU"] for r in results]

    return {
        "per_seed": results,
        "mean_mIoU": round(float(np.mean(mious)), 6),
        "std_mIoU": round(float(np.std(mious)), 6),
        "mean_FB-IoU": round(float(np.mean(fbs)), 6),
        "std_FB-IoU": round(float(np.std(fbs)), 6),
        "seeds": seeds,
    }


# ═══════════════════════════════════════════════════════════════════
# 诊断统计 | Diagnostic Statistics
# ═══════════════════════════════════════════════════════════════════

def compute_diagnostics(
    results: dict,
    support_cache: dict | None = None,
) -> dict:
    """计算 FSS 特有诊断统计 | Compute FSS-specific diagnostic stats.

    :param results: evaluate_one_fold 的输出 | output from evaluate_one_fold.
    :param support_cache: 可选的 support cache 用于分析 | optional cache for analysis.
    :return: diagnostics dict.
    """
    diag: dict = {}

    # Per-sample IoU distribution
    sample_mious = [s["mIoU"] for s in results.get("per_sample", [])]
    if sample_mious:
        diag["sample_mIoU"] = {
            "min": round(float(np.min(sample_mious)), 6),
            "q25": round(float(np.percentile(sample_mious, 25)), 6),
            "q50": round(float(np.percentile(sample_mious, 50)), 6),
            "q75": round(float(np.percentile(sample_mious, 75)), 6),
            "max": round(float(np.max(sample_mious)), 6),
        }

    # Per-class breakdown with gap analysis
    per_class = results.get("per_class_IoU", {})
    if per_class:
        valid = [(c, v["IoU"]) for c, v in per_class.items() if v["IoU"] is not None]
        valid.sort(key=lambda x: x[1])
        if valid:
            diag["best_class"] = {"id": valid[-1][0], "IoU": valid[-1][1]}
            diag["worst_class"] = {"id": valid[0][0], "IoU": valid[0][1]}
            diag["iou_gap"] = round(valid[-1][1] - valid[0][1], 6)

    # Class frequency vs IoU correlation
    if per_class:
        pairs = [(c, v["IoU"] or 0.0, v["GT_tiles"])
                 for c, v in per_class.items()]
        diag["class_stats"] = [{"class_id": c, "IoU": iou, "GT_tiles": n}
                               for c, iou, n in pairs]

    return diag


# ═══════════════════════════════════════════════════════════════════
# 输出格式化 | Output Formatting
# ═══════════════════════════════════════════════════════════════════

SEP = "=" * 70
SEP2 = "-" * 70


def _format_iou(val: float | None) -> str:
    if val is None:
        return "    N/A"
    return f"{val:8.4f}"


def print_single_fold(results: dict, fold: int, k_shot: int, mode: str, seed: int) -> None:
    """打印单 fold 评估结果 (论文风格) | Print single fold results (paper-style)."""
    print(f"\n{SEP}")
    print(f"  iSAID-5i Few-shot Segmentation Results")
    print(f"  Fold={fold}  K={k_shot}  Mode={mode}  Seed={seed}")
    print(SEP)
    print(f"  Classes:   {results['n_classes']}")
    print(f"  Tiles:     {results['n_tiles']}")
    print(SEP)
    print(f"  mIoU:      {results['mIoU']:.4f}")
    print(f"  FB-IoU:    {results['FB-IoU']:.4f}")
    print(f"  FG-IoU:    {results['FG-IoU']:.4f}")
    print(f"  BG-IoU:    {results['BG-IoU']:.4f}")

    # Novel/Base split (only for mode="all")
    if results.get("novel_mIoU") is not None:
        print(SEP2)
        print(f"  Novel mIoU:  {results['novel_mIoU']:.4f}  "
              f"(FB-IoU: {results.get('novel_FB-IoU', 0):.4f})")
        print(f"  Base  mIoU:  {results['base_mIoU']:.4f}  "
              f"(FB-IoU: {results.get('base_FB-IoU', 0):.4f})")
    print(SEP)
    print(f"  {'Class':<24s} {'IoU':>8s}  {'GT#':>6s}  {'Sup#':>5s}")
    print(SEP2)

    per_class = results.get("per_class_IoU", {})
    for cls_id in sorted(per_class):
        v = per_class[cls_id]
        name = ISAID5I_CATEGORIES.get(cls_id, f"cls{cls_id}")
        label = f"  {name} ({cls_id})"
        print(f"  {label:<24s} {_format_iou(v['IoU'])}  {v['GT_tiles']:>6d}  "
              f"{v['Support_tiles']:>5d}")
    print(SEP)


def print_cross_fold(results: dict, k_shot: int, seed: int) -> None:
    """打印多 fold 汇总 | Print cross-fold summary."""
    print(f"\n{SEP}")
    print(f"  Cross-Fold Results (K={k_shot}, Seed={seed})")
    print(SEP)

    folds = ["fold0", "fold1", "fold2"]
    available = [f for f in folds if f in results]

    print(f"  {'Metric':<16s}", end="")
    for f in available:
        print(f"  {f.capitalize():>8s}", end="")
    print(f"  {'Mean':>8s}")
    print(SEP2)

    for metric, label in [("mIoU", "mIoU"), ("FB-IoU", "FB-IoU")]:
        vals = [results[f][metric] for f in available]
        print(f"  {label:<16s}", end="")
        for v in vals:
            print(f"  {v:>8.2f}", end="")
        mean = results.get(f"mean_{metric}", float(np.mean(vals)))
        print(f"  {mean:>8.2f}")
    print(SEP)


def print_multi_seed(results: dict, fold: int, k_shot: int) -> None:
    """打印多 seed 汇总 | Print multi-seed summary."""
    seeds = results.get("seeds", [])
    print(f"\n{SEP}")
    print(f"  Multi-Seed Results (Fold={fold}, K={k_shot})")
    print(f"  Seeds: {seeds}")
    print(SEP)
    print(f"  mIoU:    {results['mean_mIoU']:.4f} ± {results['std_mIoU']:.4f}")
    print(f"  FB-IoU:  {results['mean_FB-IoU']:.4f} ± {results['std_FB-IoU']:.4f}")

    if "per_seed" in results:
        print(SEP2)
        print(f"  {'Seed':>6s}  {'mIoU':>8s}  {'FB-IoU':>8s}")
        print(SEP2)
        for i, r in enumerate(results["per_seed"]):
            print(f"  {seeds[i]:>6d}  {r['mIoU']:>8.2f}  {r['FB-IoU']:>8.2f}")
    print(SEP)


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AdaSAM iSAID-5i Few-shot Evaluation (FSS Benchmark Protocol)"
    )
    # Required
    p.add_argument("--checkpoint", required=True, help="path to checkpoint .pt file")

    # Data / Model
    p.add_argument("--data-root", default="data/iSAID-5i")
    p.add_argument("--fold", type=int, default=0, help="fold 0/1/2 (ignored with --all-folds)")
    p.add_argument("--k-shot", type=int, default=None, help="override k-shot from checkpoint")
    p.add_argument("--mode", default=None, choices=["base", "novel", "all"],
                   help="override mode from checkpoint")

    # Evaluation protocol
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42, help="primary seed for evaluation")
    p.add_argument("--score-thr", type=float, default=None,
                   help="score threshold (default: from config or 0.3)")

    # Multi-fold / Multi-seed
    p.add_argument("--all-folds", action="store_true",
                   help="run all 3 folds and report mean")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="multiple seeds for Mean±Std, e.g. --seeds 42 123 456")

    # Output options
    p.add_argument("--output-dir", default=None, help="custom output directory")
    p.add_argument("--save-predictions", action="store_true",
                   help="save all predictions as JSON")
    p.add_argument("--save-vis", action="store_true",
                   help="save visualization images")
    p.add_argument("--vis-samples", type=int, default=10, help="number of vis samples")
    p.add_argument("--diagnostics", action="store_true",
                   help="compute and output diagnostic statistics")
    p.add_argument("--max-samples", type=int, default=0,
                   help="limit evaluation to first N tiles (0=all, for smoke testing)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Determine k_shot and mode from checkpoint if not overridden
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    k_shot = args.k_shot if args.k_shot is not None else ckpt.get("k_shot", 5)
    mode = args.mode if args.mode is not None else ckpt.get("mode", "novel")
    cfg = ckpt.get("config", {})
    score_thr = (args.score_thr if args.score_thr is not None
                 else cfg.get("eval", {}).get("score_thr", 0.3))
    del ckpt  # free memory

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = _REPO_ROOT / data_root

    out_dir = (Path(args.output_dir) if args.output_dir
               else ckpt_path.parent / "eval_isaid5i")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] checkpoint: {ckpt_path}")
    print(f"[eval] k_shot={k_shot}  mode={mode}  score_thr={score_thr}")
    print(f"[eval] output: {out_dir}")

    # ── Determine evaluation mode ──
    all_folds = args.all_folds
    multi_seed = args.seeds is not None and len(args.seeds) > 1

    if all_folds and multi_seed:
        # Full: 3 folds × N seeds
        print(f"[eval] mode: all-folds × multi-seed ({len(args.seeds)} seeds)")
        folds = [0, 1, 2]
        all_results: dict[str, dict] = {}

        for fold in folds:
            seed_results = run_seeds(
                checkpoint_path=ckpt_path, data_root=data_root,
                fold=fold, mode=mode, k_shot=k_shot, seeds=args.seeds,
                score_thr=score_thr, device=device, max_samples=args.max_samples,
            )
            all_results[f"fold{fold}"] = seed_results

        # Print summary
        print(f"\n{SEP}")
        print(f"  Full Results: 3 Folds × {len(args.seeds)} Seeds")
        print(f"  K={k_shot}  Mode={mode}")
        print(SEP)
        print(f"  {'Fold':<8s}  {'mIoU':>16s}  {'FB-IoU':>18s}")
        print(SEP2)

        fold_mious = []
        fold_fbs = []
        for fold in folds:
            sr = all_results[f"fold{fold}"]
            miou_str = f"{sr['mean_mIoU']:.2f} ± {sr['std_mIoU']:.2f}"
            fb_str = f"{sr['mean_FB-IoU']:.2f} ± {sr['std_FB-IoU']:.2f}"
            print(f"  Fold {fold}    {miou_str:>16s}  {fb_str:>18s}")
            fold_mious.append(sr["mean_mIoU"])
            fold_fbs.append(sr["mean_FB-IoU"])

        print(SEP2)
        print(f"  {'Mean':<8s}  {np.mean(fold_mious):>16.2f}  {np.mean(fold_fbs):>18.2f}")
        print(SEP)

        # Save
        out_path = out_dir / "eval_results_all.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "checkpoint": str(ckpt_path), "k_shot": k_shot, "mode": mode,
                "seeds": args.seeds, "folds": folds, "results": all_results,
                "summary": {
                    "mean_mIoU": round(float(np.mean(fold_mious)), 4),
                    "mean_FB-IoU": round(float(np.mean(fold_fbs)), 4),
                },
            }, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {out_path}")

    elif all_folds:
        # 3 folds, single seed
        print(f"[eval] mode: all-folds (seed={args.seed})")
        results = run_folds(
            checkpoint_path=ckpt_path, data_root=data_root,
            folds=[0, 1, 2], mode=mode, k_shot=k_shot,
            seed=args.seed, score_thr=score_thr, device=device,
            max_samples=args.max_samples,
        )
        print_cross_fold(results, k_shot, args.seed)

        # Save (strip per_sample for compactness)
        for key in list(results):
            if key.startswith("fold"):
                results[key].pop("per_sample", None)
        out_path = out_dir / "eval_results_folds.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "checkpoint": str(ckpt_path), "k_shot": k_shot, "mode": mode,
                "seed": args.seed, "results": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {out_path}")

    elif multi_seed:
        # Single fold, multiple seeds
        print(f"[eval] mode: multi-seed fold={args.fold} ({len(args.seeds)} seeds)")
        results = run_seeds(
            checkpoint_path=ckpt_path, data_root=data_root,
            fold=args.fold, mode=mode, k_shot=k_shot,
            seeds=args.seeds, score_thr=score_thr, device=device,
            max_samples=args.max_samples,
        )
        print_multi_seed(results, args.fold, k_shot)

        # Save
        for r in results.get("per_seed", []):
            r.pop("per_sample", None)
        out_path = out_dir / "eval_results_seeds.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "checkpoint": str(ckpt_path), "fold": args.fold,
                "k_shot": k_shot, "mode": mode, "results": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {out_path}")

    else:
        # Single fold, single seed
        print(f"[eval] mode: single fold={args.fold} seed={args.seed}")
        results = evaluate_one_fold(
            checkpoint_path=ckpt_path, data_root=data_root,
            fold=args.fold, mode=mode, k_shot=k_shot,
            seed=args.seed, score_thr=score_thr, device=device,
            max_samples=args.max_samples,
        )
        print_single_fold(results, args.fold, k_shot, mode, args.seed)

        # Diagnostics
        if args.diagnostics:
            diag = compute_diagnostics(results)
            print(f"\n{SEP}")
            print(f"  Diagnostics")
            print(SEP)
            if "sample_mIoU" in diag:
                s = diag["sample_mIoU"]
                print(f"  Sample mIoU: min={s['min']:.4f} q25={s['q25']:.4f} "
                      f"q50={s['q50']:.4f} q75={s['q75']:.4f} max={s['max']:.4f}")
            if "best_class" in diag:
                best_name = ISAID5I_CATEGORIES.get(diag["best_class"]["id"], "?")
                worst_name = ISAID5I_CATEGORIES.get(diag["worst_class"]["id"], "?")
                print(f"  Best:  {best_name} ({diag['best_class']['IoU']:.4f})")
                print(f"  Worst: {worst_name} ({diag['worst_class']['IoU']:.4f})")
                print(f"  Gap:   {diag['iou_gap']:.4f}")
            print(SEP)
            results["diagnostics"] = diag

        # Save
        per_sample = results.pop("per_sample", None)
        out_path = out_dir / "eval_results.json"
        results_for_save = {
            "checkpoint": str(ckpt_path), "fold": args.fold,
            "k_shot": k_shot, "mode": mode, "seed": args.seed,
            "results": results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results_for_save, f, indent=2, ensure_ascii=False)
        print(f"  Results saved: {out_path}")

        if per_sample:
            ps_path = out_dir / "eval_per_sample.json"
            with open(ps_path, "w", encoding="utf-8") as f:
                json.dump(per_sample, f, indent=2, ensure_ascii=False)

        # Predictions JSON
        if args.save_predictions and "predictions" in results:
            preds = results.pop("predictions")
            pred_path = out_dir / "predictions.json"
            with open(pred_path, "w", encoding="utf-8") as f:
                json.dump({
                    "info": {"fold": args.fold, "k_shot": k_shot, "mode": mode,
                             "seed": args.seed, "score_thr": score_thr},
                    "predictions": preds,
                }, f, indent=2, ensure_ascii=False)
            print(f"  Predictions saved: {pred_path} ({len(preds)} entries)")

        # Visualizations
        if args.save_vis:
            _save_visualizations(results, ckpt_path, data_root, args.fold, mode,
                                 k_shot, args.seed, score_thr, device,
                                 n=args.vis_samples, out_dir=out_dir)

    print("[Done]")


# ═══════════════════════════════════════════════════════════════════
# 可视化 | Visualization
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def _save_visualizations(
    results: dict,
    checkpoint_path: Path,
    data_root: Path,
    fold: int,
    mode: str,
    k_shot: int,
    seed: int,
    score_thr: float,
    device: torch.device,
    n: int = 10,
    out_dir: Path | None = None,
) -> None:
    """Save side-by-side visualizations for random query tiles."""
    vis_dir = (out_dir or checkpoint_path.parent / "eval_isaid5i") / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Re-build model (lightweight re-creation for standalone vis)
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    bb_cfg = cfg.get("backbone", {})
    bb_path = Path(bb_cfg.get("checkpoint", "weights/mobile_sam.pt"))
    bb_path = bb_path if bb_path.is_absolute() else _REPO_ROOT / bb_path

    sam = build_mobile_sam(str(bb_path), bb_cfg.get("model_type", "vit_t"), device)
    backbone = MobileSAMBackbone(sam.image_encoder, sam.image_encoder.img_size).to(device)
    embed_dim = int(cfg.get("support_encoder", {}).get("embed_dim", 256))
    model = AdaSAMModel(sam, AdaSAMModelConfig.from_dict(cfg)).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[eval] WARNING: missing keys (new code, old ckpt): {missing}")
    if unexpected:
        print(f"[eval] WARNING: unexpected keys (new ckpt, old code): {unexpected}")
    model.eval()

    cat_adapter = None
    if ckpt.get("cat_adapter") is not None:
        tcfg = cfg.get("train", {})
        adapter_cfg = tcfg.get("cat_adapter", {})
        cat_adapter = CATAdapter(
            dim=embed_dim, bottleneck=int(adapter_cfg.get("bottleneck", 64)),
        ).to(device)
        cat_adapter.load_state_dict(ckpt["cat_adapter"])
        cat_adapter.eval()

    val_ds = ISAID5iDataset(root=str(data_root), fold=fold, split="val", mode=mode)
    visible_classes = val_ds.visible_classes()

    support_cache = build_support_cache(
        data_root=data_root, fold=fold, mode=mode, k_shot=k_shot,
        visible_classes=visible_classes, backbone=backbone,
        cat_adapter=cat_adapter, support_seed=seed, device=device,
    )

    rng = random.Random(seed + 9999)
    indices = rng.sample(range(len(val_ds)), min(n, len(val_ds)))

    # Colormap for 16 classes
    colors = np.array([
        [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 0], [255, 0, 255], [0, 255, 255], [128, 0, 0],
        [0, 128, 0], [0, 0, 128], [128, 128, 0], [128, 0, 128],
        [0, 128, 128], [192, 192, 192], [64, 64, 64], [255, 128, 0],
    ], dtype=np.uint8)

    for idx in tqdm(indices, desc="visualize"):
        sample = val_ds[idx]
        query_image = sample["image"]
        H, W = query_image.shape[1], query_image.shape[2]
        img_np = (query_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Embed query
        x, meta = preprocess_image(query_image)
        query_emb = backbone(x.unsqueeze(0).to(device))["image_embedding"]
        if cat_adapter is not None:
            query_emb = cat_adapter(query_emb)

        # GT (semantic: class-level merge from dataset)
        gt_combined = np.zeros((H, W), dtype=np.uint8)
        for cls in visible_classes:
            gt_m = val_ds.get_class_mask(idx, cls)
            if gt_m is not None and gt_m.sum() > 0:
                gt_combined[gt_m.numpy().astype(bool)] = cls

        # Predict each visible class
        pred_combined = np.zeros((H, W), dtype=np.uint8)
        for cls in visible_classes:
            cached = support_cache.get(cls)
            if cached is None:
                continue
            support_features, support_masks_grid = cached
            masks, scores = model.predict(
                query_features=query_emb,
                support_features=support_features,
                support_masks=support_masks_grid,
                input_size=meta.input_size,
                original_size=(H, W),
                score_thr=score_thr,
            )
            if masks.shape[0] > 0:
                pred_combined[masks[0].cpu().numpy()] = cls

        # Colorize
        gt_col = _colorize(gt_combined, colors)
        pred_col = _colorize(pred_combined, colors)

        # Diff: gray=correct, red=FP, blue=FN
        diff = np.zeros((H, W, 3), dtype=np.uint8)
        correct = (gt_combined == pred_combined)
        diff[correct] = (128, 128, 128)
        fp = (pred_combined > 0) & (gt_combined == 0)
        diff[fp] = (255, 0, 0)
        fn = (gt_combined > 0) & (pred_combined == 0)
        diff[fn] = (0, 0, 255)

        combined = np.hstack([img_np, gt_col, pred_col, diff])
        cv2.imwrite(
            str(vis_dir / f"{sample.get('tile_id', idx)}.png"),
            cv2.cvtColor(combined, cv2.COLOR_RGB2BGR),
        )

    print(f"  Visualizations saved to: {vis_dir}")


def _colorize(label: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Convert class label map to color image using given colormap."""
    H, W = label.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for c in range(len(colors)):
        out[label == c] = colors[c]
    return out


if __name__ == "__main__":
    main()
