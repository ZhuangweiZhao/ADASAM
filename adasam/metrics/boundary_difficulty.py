"""Streaming metrics for boundary-difficulty signal diagnostics."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def semantic_boundary(mask: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """Return one-pixel semantic boundaries for integer masks shaped [B,H,W]."""
    if mask.ndim != 3:
        raise ValueError("mask must have shape [B,H,W]")
    valid = mask != ignore_index
    boundary = torch.zeros_like(valid)
    horizontal = valid[:, :, 1:] & valid[:, :, :-1] & (mask[:, :, 1:] != mask[:, :, :-1])
    vertical = valid[:, 1:, :] & valid[:, :-1, :] & (mask[:, 1:, :] != mask[:, :-1, :])
    boundary[:, :, 1:] |= horizontal
    boundary[:, :, :-1] |= horizontal
    boundary[:, 1:, :] |= vertical
    boundary[:, :-1, :] |= vertical
    return boundary


def boundary_band(mask: torch.Tensor, radius: int, ignore_index: int = 255) -> torch.Tensor:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    boundary = semantic_boundary(mask, ignore_index)
    if radius == 0:
        return boundary
    kernel = 2 * radius + 1
    return F.max_pool2d(boundary[:, None].float(), kernel, stride=1, padding=radius)[:, 0].bool()


class HistogramBinaryMetrics:
    """Bounded-memory approximation of binary ranking metrics for scores in [0,1]."""

    def __init__(self, bins: int = 2048) -> None:
        if bins < 16:
            raise ValueError("bins must be at least 16")
        self.bins = bins
        self.positive = np.zeros(bins, dtype=np.int64)
        self.negative = np.zeros(bins, dtype=np.int64)

    def update(self, score: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> None:
        score_np = score[valid].detach().float().clamp(0, 1).cpu().numpy()
        target_np = target[valid].detach().bool().cpu().numpy()
        index = np.minimum((score_np * self.bins).astype(np.int64), self.bins - 1)
        self.positive += np.bincount(index[target_np], minlength=self.bins)
        self.negative += np.bincount(index[~target_np], minlength=self.bins)

    def compute(self, top_fractions: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3)) -> dict:
        positives = int(self.positive.sum())
        negatives = int(self.negative.sum())
        total = positives + negatives
        if positives == 0 or negatives == 0:
            return {"positives": positives, "negatives": negatives, "pr_auc": None,
                    "roc_auc": None, "prevalence": positives / total if total else None,
                    "top_fraction": {}}
        tp = np.cumsum(self.positive[::-1], dtype=np.float64)
        fp = np.cumsum(self.negative[::-1], dtype=np.float64)
        recall = tp / positives
        precision = tp / np.maximum(tp + fp, 1)
        previous_recall = np.r_[0.0, recall[:-1]]
        average_precision = float(np.sum((recall - previous_recall) * precision))
        tpr = np.r_[0.0, recall]
        fpr = np.r_[0.0, fp / negatives]
        roc_auc = float(np.trapezoid(tpr, fpr))
        prevalence = positives / total
        ranked_total = tp + fp
        top = {}
        for fraction in top_fractions:
            cutoff = max(1, int(np.ceil(total * fraction)))
            idx = int(np.searchsorted(ranked_total, cutoff, side="left"))
            idx = min(idx, len(tp) - 1)
            error_rate = float(tp[idx] / max(ranked_total[idx], 1))
            top[f"{fraction:.2f}"] = {
                "selected_fraction_approx": float(ranked_total[idx] / total),
                "positive_rate": error_rate,
                "enrichment": float(error_rate / prevalence),
            }
        return {"positives": positives, "negatives": negatives,
                "prevalence": float(prevalence), "pr_auc": average_precision,
                "roc_auc": roc_auc, "top_fraction": top}
