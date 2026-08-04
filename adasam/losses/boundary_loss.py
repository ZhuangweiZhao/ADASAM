"""Ignore-aware semantic boundary targets, loss, and evaluation metrics."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _boundaries(labels: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    boundary = torch.zeros_like(valid)
    horizontal_valid = valid[:, :, 1:] & valid[:, :, :-1]
    horizontal = (labels[:, :, 1:] != labels[:, :, :-1]) & horizontal_valid
    boundary[:, :, 1:] |= horizontal
    boundary[:, :, :-1] |= horizontal
    vertical_valid = valid[:, 1:, :] & valid[:, :-1, :]
    vertical = (labels[:, 1:, :] != labels[:, :-1, :]) & vertical_valid
    boundary[:, 1:, :] |= vertical
    boundary[:, :-1, :] |= vertical
    return boundary & valid, valid


def semantic_boundary_target(
    target: torch.Tensor, ignore_index: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return binary boundaries and valid pixels from a semantic target [B,H,W]."""
    if target.ndim != 3:
        raise ValueError("target must have shape [B,H,W]")
    valid = torch.ones_like(target, dtype=torch.bool)
    if ignore_index is not None:
        valid = target != ignore_index
    return _boundaries(target, valid)


class BoundaryLoss(nn.Module):
    """Class-balanced BCE plus Dice for a one-channel boundary prediction."""

    def __init__(self, ignore_index: int | None = None, eps: float = 1e-6) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError("boundary logits must have shape [B,1,H,W]")
        if logits.shape[0] != target.shape[0]:
            raise ValueError("boundary logits and target batch sizes must match")
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(logits, target.shape[-2:], mode="bilinear", align_corners=False)
        boundary, valid = semantic_boundary_target(target, self.ignore_index)
        labels = boundary.unsqueeze(1).to(logits.dtype)
        valid_f = valid.unsqueeze(1).to(logits.dtype)
        positives = (labels * valid_f).sum()
        negatives = ((1.0 - labels) * valid_f).sum()
        pos_weight = (negatives / positives.clamp_min(1.0)).clamp(max=50.0)
        bce = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pos_weight, reduction="none"
        )
        bce = (bce * valid_f).sum() / valid_f.sum().clamp_min(1.0)
        probability = logits.sigmoid() * valid_f
        intersection = (probability * labels).sum()
        dice = 1.0 - (2.0 * intersection + self.eps) / (
            probability.sum() + labels.sum() + self.eps
        )
        return bce + dice


def boundary_f1_counts(
    prediction: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int | None = None,
    tolerance: int = 2,
) -> tuple[int, int, int, int]:
    """Return matched-pred, predicted, matched-target, target boundary pixel counts."""
    valid = torch.ones_like(target, dtype=torch.bool)
    if ignore_index is not None:
        valid = target != ignore_index
    pred_boundary, _ = _boundaries(prediction, valid)
    target_boundary, _ = _boundaries(target, valid)
    kernel = 2 * max(tolerance, 0) + 1
    pred_f = pred_boundary.unsqueeze(1).float()
    target_f = target_boundary.unsqueeze(1).float()
    if kernel > 1:
        dilated_pred = F.max_pool2d(pred_f, kernel, stride=1, padding=tolerance).bool()
        dilated_target = F.max_pool2d(target_f, kernel, stride=1, padding=tolerance).bool()
    else:
        dilated_pred = pred_f.bool()
        dilated_target = target_f.bool()
    matched_pred = int((pred_f.bool() & dilated_target).sum())
    matched_target = int((target_f.bool() & dilated_pred).sum())
    return matched_pred, int(pred_f.sum()), matched_target, int(target_f.sum())
