"""
实例级匹配与 Instance mIoU | Instance-level Matching & Instance mIoU.
=====================================================================

严格实例分割评估的核心度量 (纯 numpy, 无 pycocotools 依赖, 便于单测).
Core metrics for strict instance segmentation evaluation
(pure numpy, no pycocotools dependency, easy to unit-test).

设计原则 | Design principles:
    1. 实例级, 绝不 union | Instance-level, never union masks.
       每个 GT 实例、每个预测实例都是独立个体.
       Each GT instance and each prediction is a separate entity.
    2. 一对一匹配 | One-to-one matching.
       一个 prediction 不能同时匹配多个 GT (贪心, 按 score 降序).
       A prediction cannot match multiple GT (greedy, by descending score).
    3. Instance mIoU 按需求定义 | Instance mIoU as specified:
       对每个 GT 实例取最大 IoU 预测, 再对所有 GT 求平均.
       For each GT, take the max-IoU prediction, then average over all GT.

与 COCO AP 的关系 | Relationship with COCO AP:
    AP 由官方 pycocotools 计算 (见 adasam/metrics/coco_eval.py).
    本模块提供 AP 之外的补充度量: Instance mIoU + TP/FP/FN 调试计数.
    AP is computed by official pycocotools (see coco_eval.py).
    This module provides supplementary metrics: Instance mIoU + TP/FP/FN debug counts.
"""

from __future__ import annotations

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# IoU 矩阵 | Pairwise IoU matrix
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

    IoU(p, g) = |p ∩ g| / |p ∪ g|, 逐实例, 不做任何 union 合并.
    Per-instance IoU, no union merging whatsoever.

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
# 贪心一对一匹配 | Greedy one-to-one matching
# ═══════════════════════════════════════════════════════════════════

def greedy_match(pred_masks: list[np.ndarray],
                 pred_scores: list[float] | np.ndarray,
                 gt_masks: list[np.ndarray],
                 iou_thr: float = 0.5) -> dict:
    """贪心一对一匹配 → TP/FP/FN | Greedy one-to-one matching → TP/FP/FN.

    流程 | Procedure:
        1. 预测按 score 降序排列 | Sort predictions by descending score.
        2. 每个预测在"尚未被匹配"的 GT 中选 IoU 最大且 ≥ 阈值者认领.
           Each prediction claims the highest-IoU *still-unmatched* GT with IoU ≥ thr.
        3. 一个 prediction 最多匹配一个 GT, 一个 GT 最多被匹配一次.
           One prediction ↔ at most one GT; each GT matched at most once.

    这是需求 #3 的严格实现 (禁止一个 prediction 匹配多个 GT).
    This strictly implements requirement #3 (no prediction matches multiple GT).

    :param pred_masks: P 个预测掩码 | P prediction masks.
    :param pred_scores: P 个置信度 | P confidence scores.
    :param gt_masks: G 个 GT 掩码 | G GT masks.
    :param iou_thr: 匹配 IoU 阈值 | IoU threshold for a valid match.
    :return: dict with:
        - "matched_pairs": [(pred_idx, gt_idx, iou), ...]
        - "tp": 真正例数 (成功匹配的预测) | True positives.
        - "fp": 假正例数 (未匹配的预测) | False positives.
        - "fn": 假负例数 (未匹配的 GT) | False negatives.
        - "n_pred": 预测总数 | Number of predictions.
        - "n_gt": GT 总数 | Number of GT.
    """
    P, G = len(pred_masks), len(gt_masks)

    # ── 边界情况 | Edge cases ──
    if P == 0:
        return {"matched_pairs": [], "tp": 0, "fp": 0, "fn": G, "n_pred": 0, "n_gt": G}
    if G == 0:
        return {"matched_pairs": [], "tp": 0, "fp": P, "fn": 0, "n_pred": P, "n_gt": 0}

    iou = pairwise_iou(pred_masks, gt_masks)      # [P, G]
    scores = np.asarray(pred_scores, dtype=np.float32)

    # 预测按 score 降序 | Predictions in descending score order
    order = np.argsort(-scores, kind="stable")

    gt_taken = np.zeros(G, dtype=bool)            # GT 是否已被认领 | GT claimed?
    matched_pairs = []
    tp = 0
    for p in order:
        # 仅在未被认领的 GT 中挑 | Choose only among unclaimed GT
        candidate_ious = iou[p].copy()
        candidate_ious[gt_taken] = -1.0           # 屏蔽已认领 | Mask claimed
        g = int(np.argmax(candidate_ious))
        best_iou = float(candidate_ious[g])
        if best_iou >= iou_thr:
            gt_taken[g] = True
            matched_pairs.append((int(p), g, best_iou))
            tp += 1
        # 否则该预测为 FP (不认领任何 GT) | Otherwise this prediction is FP

    fp = P - tp
    fn = G - tp
    return {
        "matched_pairs": matched_pairs,
        "tp": tp, "fp": fp, "fn": fn,
        "n_pred": P, "n_gt": G,
    }


# ═══════════════════════════════════════════════════════════════════
# Instance mIoU (需求 #5 定义) | Instance mIoU (requirement #5 definition)
# ═══════════════════════════════════════════════════════════════════

def instance_miou(pred_masks: list[np.ndarray],
                  gt_masks: list[np.ndarray]) -> tuple[list[float], float]:
    """Instance mIoU: 每个 GT 取最大 IoU 预测, 再对所有 GT 求平均.
    Instance mIoU: per GT the max-IoU prediction, averaged over all GT.

    注意 | Note:
        此定义 **不** 强制一对一 (两个 GT 可命中同一预测), 与 greedy_match 不同.
        This definition does NOT enforce one-to-one (two GT may pick the same
        prediction), unlike greedy_match. 二者刻意分离, 见 docs/metrics_instance_v3.md.
        The two are intentionally separate; see docs/metrics_instance_v3.md.

    绝不使用 union mask | Never uses a union mask.

    :param pred_masks: P 个预测掩码 | P prediction masks.
    :param gt_masks: G 个 GT 掩码 | G GT masks.
    :return: (per_gt_max_iou, mean)
        - per_gt_max_iou: 长度 G 的列表, 每个 GT 的最大 IoU | list of length G.
        - mean: 上述平均值 (G==0 时为 NaN, 由调用方跳过) | mean (NaN if G==0).
    """
    G = len(gt_masks)
    if G == 0:
        return [], float("nan")            # 无 GT → 该图不计入 | No GT → skip this image
    if len(pred_masks) == 0:
        return [0.0] * G, 0.0              # 有 GT 无预测 → 全 0 | GT but no pred → all zero

    iou = pairwise_iou(pred_masks, gt_masks)   # [P, G]
    per_gt_max = iou.max(axis=0)               # 每个 GT 的最大 IoU | max over predictions
    per_gt_list = [float(x) for x in per_gt_max]
    return per_gt_list, float(np.mean(per_gt_max))
