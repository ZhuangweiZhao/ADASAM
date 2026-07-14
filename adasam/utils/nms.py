"""
Mask IoU NMS | 掩码 IoU 非极大值抑制.
======================================

基于掩码 IoU 的 NMS, 使用 adasam.metrics.instance_match.pairwise_iou 计算 IoU 矩阵。
Mask IoU-based NMS, using adasam.metrics.instance_match.pairwise_iou for IoU computation.

用法 | Usage::

    from adasam.utils.nms import mask_iou_nms
    keep = mask_iou_nms(masks, scores, iou_threshold=0.6)
    masks_nms = masks[keep]
    scores_nms = scores[keep]
"""

from __future__ import annotations

import torch

from adasam.metrics.instance_match import pairwise_iou


def mask_iou_nms(
    masks: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.6,
) -> torch.Tensor:
    """掩码 IoU NMS | Mask IoU NMS.

    按 score 降序排列预测, 贪婪抑制与已保留预测 IoU > iou_threshold 的掩码。
    Sorts predictions by descending score, greedily suppresses masks whose IoU
    with any higher-score kept prediction exceeds iou_threshold.

    :param masks: [N, H, W] bool 二值掩码 | bool binary masks.
    :param scores: [N] float 置信度 ∈ [0, 1] | confidence scores.
    :param iou_threshold: IoU 抑制阈值 | IoU suppression threshold.
    :return: [M] LongTensor — 保留的索引 | indices to keep.
    """
    if masks.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=masks.device)

    if masks.shape[0] == 1:
        return torch.tensor([0], dtype=torch.long, device=masks.device)

    # Sort by score descending
    order = torch.argsort(scores, descending=True)
    masks_sorted = masks[order]  # [N, H, W]

    keep: list[int] = []
    suppressed = torch.zeros(masks.shape[0], dtype=torch.bool, device=masks.device)

    for i in range(masks_sorted.shape[0]):
        if suppressed[i]:
            continue
        keep.append(int(order[i].item()))

        # Suppress remaining predictions with high IoU
        if i + 1 < masks_sorted.shape[0]:
            iou = pairwise_iou(
                masks_sorted[i+1:].float(),
                masks_sorted[i:i+1].float(),
            )[:, 0]  # [remaining]
            high_iou = iou > iou_threshold
            # Map back to sorted indices
            rem_indices = torch.arange(i + 1, masks_sorted.shape[0], device=masks.device)
            suppressed[rem_indices[high_iou]] = True

    return torch.tensor(keep, dtype=torch.long, device=masks.device)


def mask_iou_nms_batch(
    masks: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """掩码 IoU NMS 并返回过滤结果 | Mask IoU NMS and return filtered results.

    便捷封装: 调用 mask_iou_nms 并直接返回过滤后的 masks 和 scores。
    Convenience wrapper: calls mask_iou_nms and returns filtered masks and scores.

    :return: (masks_kept [M,H,W], scores_kept [M]).
    """
    keep = mask_iou_nms(masks, scores, iou_threshold)
    return masks[keep], scores[keep]
