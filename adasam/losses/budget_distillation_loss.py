"""Loss for predicting Magnitude top-budget masks from the semantic anchor."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MagnitudeTeacherDistillationLoss(nn.Module):
    """Class-balanced BCE plus soft Dice on the teacher's binary Top-K masks."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, student_logits: torch.Tensor, teacher_masks: torch.Tensor) -> torch.Tensor:
        if student_logits.shape != teacher_masks.shape or student_logits.ndim != 4:
            raise ValueError("student_logits and teacher_masks must share shape [B,2,H,W]")
        targets = teacher_masks.to(student_logits.dtype)
        positives = targets.sum()
        negatives = targets.numel() - positives
        pos_weight = (negatives / positives.clamp_min(1.0)).detach()
        bce = F.binary_cross_entropy_with_logits(
            student_logits, targets, pos_weight=pos_weight
        )
        probabilities = student_logits.sigmoid()
        intersection = (probabilities * targets).sum(dim=(-2, -1))
        dice = 1.0 - (
            (2.0 * intersection + self.eps)
            / (probabilities.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1)) + self.eps)
        ).mean()
        return bce + dice
