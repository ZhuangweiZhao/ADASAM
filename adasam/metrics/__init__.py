"""
adasam.metrics — 语义分割评测指标 | Semantic Segmentation Evaluation Metrics.
=============================================================================

纯 numpy 语义分割度量, 无外部 COCO API 依赖:
Pure-numpy semantic segmentation metrics, no COCO API dependency:

    - mIoU: 各类 IoU 均值 | mean IoU across classes
    - FB-IoU: 前景-背景 IoU (FSS 标准) | foreground-background IoU (FSS standard)
    - Pixel Accuracy: 像素正确率 | pixel-wise accuracy
    - pairwise_iou: 通用成对 IoU 矩阵 | generic pairwise IoU matrix
"""

from adasam.metrics.semantic_metrics import (
    pairwise_iou,
    compute_miou,
    compute_fb_iou,
    compute_fb_iou_from_accum,
    compute_pixel_accuracy,
)

__all__ = [
    "pairwise_iou",
    "compute_miou",
    "compute_fb_iou",
    "compute_fb_iou_from_accum",
    "compute_pixel_accuracy",
]
