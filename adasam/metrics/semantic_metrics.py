"""
语义分割度量 | Semantic Segmentation Metrics.
===============================================

纯 numpy 语义分割评估核心度量 (无外部 COCO API 依赖):
Core semantic segmentation evaluation metrics (pure numpy, no COCO API):

    - mIoU: 各类 IoU 的均值 | mean IoU across all classes
    - FB-IoU: 前景-背景 IoU (FSS 标准指标) | foreground-background IoU
    - Pixel Accuracy: 像素正确率 | pixel-wise accuracy
    - pairwise_iou: 通用成对 IoU 矩阵 | generic pairwise IoU matrix

设计原则 | Design principles:
    1. 类级别评估, 同类实例合并为单一语义区域。
       Class-level evaluation; same-class instances merged into one semantic region.
    2. 背景类 (cls=0) 作为独立类别参与计算。
       Background (cls=0) is treated as an independent class.
    3. mIoU = mean(IoU_c) for c in visible_classes.
"""

from __future__ import annotations

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# IoU 矩阵 | Pairwise IoU matrix (通用 | generic)
# ═══════════════════════════════════════════════════════════════════

def _stack_masks(masks: list[np.ndarray]) -> np.ndarray:
    """将 mask 列表堆叠为 [N, H*W] 的 float32 平面数组 | Stack masks to [N, H*W] float32.

    :param masks: N 个 [H, W] bool/uint8 二值掩码 | N binary masks.
    :return: [N, H*W] float32 (空列表 → [0, 0]) | [N, H*W] float32 (empty → [0, 0]).
    """
    if len(masks) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    flat = [np.asarray(m, dtype=bool).reshape(-1).astype(np.float32) for m in masks]
    return np.stack(flat, axis=0)  # [N, H*W]


def pairwise_iou(pred_masks: list[np.ndarray],
                 gt_masks: list[np.ndarray]) -> np.ndarray:
    """计算预测与 GT 的成对 IoU 矩阵 | Compute pairwise IoU matrix between predictions and GT.

    IoU(p, g) = |p ∩ g| / |p ∪ g|, 逐掩码, 不做任何 union 合并。
    Per-mask IoU, no union merging whatsoever.

    :param pred_masks: P 个预测掩码 [H, W] | P prediction masks.
    :param gt_masks: G 个 GT 掩码 [H, W] | G GT masks.
    :return: [P, G] float32 IoU 矩阵 (P 或 G 为 0 时返回相应空形状).
             [P, G] float32 IoU matrix (empty shape if P or G is 0).
    """
    P, G = len(pred_masks), len(gt_masks)
    if P == 0 or G == 0:
        return np.zeros((P, G), dtype=np.float32)

    pred_flat = _stack_masks(pred_masks)          # [P, HW]
    gt_flat = _stack_masks(gt_masks)              # [G, HW]

    # 交集 = 布尔与的计数 = 矩阵乘 | Intersection = count of AND = matmul
    inter = pred_flat @ gt_flat.T                 # [P, G]
    pred_area = pred_flat.sum(axis=1, keepdims=True)   # [P, 1]
    gt_area = gt_flat.sum(axis=1, keepdims=True).T     # [1, G]
    union = pred_area + gt_area - inter           # [P, G]

    # 避免除零 | Avoid divide-by-zero
    iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    return iou.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# 语义分割指标 | Semantic Segmentation Metrics
# ═══════════════════════════════════════════════════════════════════

def compute_miou(
    per_class_inter: dict[int, float],
    per_class_union: dict[int, float],
    visible_classes: list[int] | None = None,
) -> dict:
    """计算 mIoU | Compute mean IoU.

    对每个可见类别计算 IoU_c = inter_c / union_c, 然后对所有有效类别取均值。
    For each visible class, IoU_c = inter_c / union_c; then mean over all valid classes.

    :param per_class_inter: {class_id: intersection_pixels} 累积字典.
    :param per_class_union: {class_id: union_pixels} 累积字典.
    :param visible_classes: 需要计入的类别列表 (None=所有有 union 的类).
    :return: {
        "mIoU": float,
        "per_class_IoU": {cls: float or None},
        "valid_classes": int,
    }
    """
    classes = visible_classes if visible_classes is not None else list(per_class_union.keys())
    per_class = {}
    valid_ious = []

    for cls in classes:
        u = per_class_union.get(cls, 0.0)
        inter = per_class_inter.get(cls, 0.0)
        if u > 0:
            iou_c = inter / u
            per_class[cls] = iou_c
            valid_ious.append(iou_c)
        else:
            per_class[cls] = None

    miou = float(np.mean(valid_ious)) if valid_ious else 0.0
    return {
        "mIoU": round(miou, 6),
        "per_class_IoU": per_class,
        "valid_classes": len(valid_ious),
    }


def compute_fb_iou(
    per_class_inter: dict[int, float],
    per_class_union: dict[int, float],
    fg_classes: list[int],
    bg_class: int = 0,
) -> dict:
    """计算 FB-IoU | Compute Foreground-Background IoU.

    FSS 标准指标: 将所有前景类合并为 FG, 与 BG 计算 IoU。
    Standard FSS metric: merge all FG classes, compute IoU against BG.

    FB-IoU = (FG-IoU + BG-IoU) / 2

    :param per_class_inter: {class_id: intersection_pixels} 累积字典.
    :param per_class_union: {class_id: union_pixels} 累积字典.
    :param fg_classes: 前景类别 ID 列表.
    :param bg_class: 背景类别 ID (默认 0).
    :return: {"FB-IoU": float, "FG-IoU": float, "BG-IoU": float}
    """
    fg_inter = sum(per_class_inter.get(c, 0.0) for c in fg_classes)
    fg_union = sum(per_class_union.get(c, 0.0) for c in fg_classes)
    bg_inter = per_class_inter.get(bg_class, 0.0)
    bg_union = per_class_union.get(bg_class, 0.0)

    fg_iou = fg_inter / fg_union if fg_union > 0 else float("nan")
    bg_iou = bg_inter / bg_union if bg_union > 0 else float("nan")

    valid = [v for v in [fg_iou, bg_iou] if not np.isnan(v)]
    fb_iou = float(np.mean(valid)) if valid else float("nan")

    return {
        "FB-IoU": round(fb_iou, 6),
        "FG-IoU": round(fg_iou, 6),
        "BG-IoU": round(bg_iou, 6),
    }


def compute_fb_iou_from_accum(
    fg_inter: float, fg_union: float,
    bg_inter: float, bg_union: float,
) -> dict:
    """从已累积的 FG/BG inter/union 计算 FB-IoU | Compute FB-IoU from accumulated values."""
    fg_iou = fg_inter / fg_union if fg_union > 0 else float("nan")
    bg_iou = bg_inter / bg_union if bg_union > 0 else float("nan")
    valid = [v for v in [fg_iou, bg_iou] if not np.isnan(v)]
    fb_iou = float(np.mean(valid)) if valid else float("nan")
    return {
        "FB-IoU": round(fb_iou, 6),
        "FG-IoU": round(fg_iou, 6),
        "BG-IoU": round(bg_iou, 6),
    }


def compute_pixel_accuracy(
    pred: np.ndarray,
    gt: np.ndarray,
    ignore_index: int = 255,
) -> float:
    """计算像素准确率 | Compute pixel accuracy.

    :param pred: [H, W] 预测类别标签 | predicted class labels.
    :param gt: [H, W] GT 类别标签 | GT class labels.
    :param ignore_index: 忽略的标签值 | ignored label value.
    :return: pixel accuracy ∈ [0, 1].
    """
    mask = gt != ignore_index
    if mask.sum() == 0:
        return 1.0
    return float((pred[mask] == gt[mask]).sum() / mask.sum())
