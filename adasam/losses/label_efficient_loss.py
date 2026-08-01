"""Losses for the label-efficient semantic segmentation baseline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelEfficientSegmentationLoss(nn.Module):
    """Weighted multi-class cross entropy and soft Dice loss."""

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        include_background: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.include_background = include_background
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.ndim != 4 or target.ndim != 3:
            raise ValueError("expected prediction [B,C,H,W] and target [B,H,W]")
        if prediction.shape[0] != target.shape[0] or prediction.shape[-2:] != target.shape[-2:]:
            raise ValueError("prediction and target batch/spatial shapes must match")
        ce = F.cross_entropy(prediction, target.long())
        probabilities = prediction.softmax(dim=1)
        one_hot = F.one_hot(target.long(), prediction.shape[1]).permute(0, 3, 1, 2)
        one_hot = one_hot.to(probabilities.dtype)
        if not self.include_background:
            probabilities = probabilities[:, 1:]
            one_hot = one_hot[:, 1:]
        dims = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dims)
        denominator = probabilities.sum(dims) + one_hot.sum(dims)
        dice = 1.0 - ((2.0 * intersection + self.eps) / (denominator + self.eps)).mean()
        return self.ce_weight * ce + self.dice_weight * dice
