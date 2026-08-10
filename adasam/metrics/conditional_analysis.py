"""Conditioned accuracy and routing analyses for semantic segmentation."""

from __future__ import annotations

import cv2
import numpy as np
import torch


SIZE_NAMES = ("small", "medium", "large")


def component_size_map(
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
    small_max_fraction: float = 0.001,
    medium_max_fraction: float = 0.01,
) -> torch.Tensor:
    """Assign foreground pixels to small/medium/large connected-component bands.

    Returns int8 values: -1 for pixels outside foreground components, then 0/1/2.
    Thresholds are fractions of the resized image area and use 8-connectivity.
    """
    if target.ndim != 3:
        raise ValueError("target must have shape [B,H,W]")
    output = torch.full_like(target, -1, dtype=torch.int8, device="cpu")
    target_cpu = target.detach().cpu().numpy()
    image_area = target.shape[-2] * target.shape[-1]
    thresholds = (small_max_fraction * image_area, medium_max_fraction * image_area)
    for batch_index, labels in enumerate(target_cpu):
        for class_id in range(1, num_classes):
            binary = (labels == class_id).astype(np.uint8)
            count, components, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
            for component_id in range(1, count):
                area = int(stats[component_id, cv2.CC_STAT_AREA])
                band = 0 if area <= thresholds[0] else 1 if area <= thresholds[1] else 2
                output[batch_index][torch.from_numpy(components == component_id)] = band
    if ignore_index is not None:
        output[target.detach().cpu() == ignore_index] = -1
    return output


def component_iou_sums(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
    small_max_fraction: float = 0.001,
    medium_max_fraction: float = 0.01,
    padding: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return IoU sum/count per size band over GT semantic components.

    Each component is evaluated against same-class predictions in its padded bounding
    box. This is a region-conditioned diagnostic, not COCO instance AP.
    """
    sums = torch.zeros(3, dtype=torch.float64)
    counts = torch.zeros(3, dtype=torch.int64)
    pred_np = prediction.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    height, width = target.shape[-2:]
    image_area = height * width
    thresholds = (small_max_fraction * image_area, medium_max_fraction * image_area)
    for pred_labels, gt_labels in zip(pred_np, target_np):
        valid = np.ones_like(gt_labels, dtype=bool)
        if ignore_index is not None:
            valid = gt_labels != ignore_index
        for class_id in range(1, num_classes):
            count, components, stats, _ = cv2.connectedComponentsWithStats(
                (gt_labels == class_id).astype(np.uint8), 8
            )
            for component_id in range(1, count):
                x, y, w, h, area = (int(v) for v in stats[component_id])
                band = 0 if area <= thresholds[0] else 1 if area <= thresholds[1] else 2
                x0, y0 = max(0, x - padding), max(0, y - padding)
                x1, y1 = min(width, x + w + padding), min(height, y + h + padding)
                gt_component = components[y0:y1, x0:x1] == component_id
                pred_component = (pred_labels[y0:y1, x0:x1] == class_id) & valid[y0:y1, x0:x1]
                intersection = np.logical_and(gt_component, pred_component).sum()
                union = np.logical_or(gt_component, pred_component).sum()
                sums[band] += float(intersection / max(union, 1))
                counts[band] += 1
    return sums, counts


def summarize_component_iou(sums: torch.Tensor, counts: torch.Tensor) -> dict:
    return {
        name: (float(sums[index] / counts[index]) if counts[index] else None)
        for index, name in enumerate(SIZE_NAMES)
    } | {"component_counts": {name: int(counts[index]) for index, name in enumerate(SIZE_NAMES)}}
