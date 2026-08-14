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
        ignore_index: int | None = None,
        class_weights: torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.include_background = include_background
        self.ignore_index = ignore_index
        self.register_buffer("class_weights", class_weights)
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.ndim != 4 or target.ndim != 3:
            raise ValueError("expected prediction [B,C,H,W] and target [B,H,W]")
        if prediction.shape[0] != target.shape[0] or prediction.shape[-2:] != target.shape[-2:]:
            raise ValueError("prediction and target batch/spatial shapes must match")
        ce = F.cross_entropy(
            prediction,
            target.long(),
            weight=self.class_weights,
            ignore_index=self.ignore_index if self.ignore_index is not None else -100,
        )
        probabilities = prediction.softmax(dim=1)
        valid = torch.ones_like(target, dtype=torch.bool)
        if self.ignore_index is not None:
            valid = target != self.ignore_index
        safe_target = target.long().masked_fill(~valid, 0)
        one_hot = F.one_hot(safe_target, prediction.shape[1]).permute(0, 3, 1, 2)
        one_hot = one_hot.to(probabilities.dtype)
        valid_mask = valid.unsqueeze(1).to(probabilities.dtype)
        probabilities = probabilities * valid_mask
        one_hot = one_hot * valid_mask
        if not self.include_background:
            probabilities = probabilities[:, 1:]
            one_hot = one_hot[:, 1:]
        dims = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dims)
        denominator = probabilities.sum(dims) + one_hot.sum(dims)
        dice = 1.0 - ((2.0 * intersection + self.eps) / (denominator + self.eps)).mean()
        return self.ce_weight * ce + self.dice_weight * dice


class DefectPromptAlignmentLoss(nn.Module):
    """Align dense prompt activation with foreground defect regions."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, dense_prompt: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if dense_prompt.ndim != 4 or target.ndim != 3:
            raise ValueError("expected dense_prompt [B,C,H,W] and target [B,H,W]")
        if dense_prompt.shape[0] != target.shape[0]:
            raise ValueError("dense_prompt and target batch sizes must match")
        activation = dense_prompt.abs().mean(dim=1, keepdim=True)
        foreground = (target > 0).to(dtype=activation.dtype).unsqueeze(1)
        foreground = F.interpolate(
            foreground, size=activation.shape[-2:], mode="nearest"
        )
        bce = F.binary_cross_entropy_with_logits(activation, foreground)
        probability = activation.sigmoid()
        dims = (0, 2, 3)
        intersection = (probability * foreground).sum(dims)
        denominator = probability.sum(dims) + foreground.sum(dims)
        dice = 1.0 - ((2.0 * intersection + self.eps) / (denominator + self.eps)).mean()
        return self.bce_weight * bce + self.dice_weight * dice


class PrototypeCompactnessLoss(nn.Module):
    """Pull labeled feature pixels toward their corresponding class prototype."""

    def forward(
        self,
        feature: torch.Tensor,
        target: torch.Tensor,
        prototypes: torch.Tensor,
        initialized: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if feature.ndim != 4 or target.ndim != 3 or prototypes.ndim != 2:
            raise ValueError("expected feature [B,D,H,W], target [B,H,W], prototypes [C,D]")
        target_small = F.interpolate(
            target.unsqueeze(1).float(), size=feature.shape[-2:], mode="nearest"
        ).squeeze(1).long()
        class_losses = []
        for class_id in range(prototypes.shape[0]):
            if initialized is not None and not bool(initialized[class_id]):
                continue
            class_mask = target_small == class_id
            if not class_mask.any():
                continue
            pixels = feature.permute(0, 2, 3, 1)[class_mask]
            class_prototype = prototypes[class_id].detach().unsqueeze(0)
            class_losses.append(
                (1.0 - F.cosine_similarity(pixels, class_prototype, dim=1, eps=1e-6)).mean()
            )
        if not class_losses:
            return feature.sum() * 0.0
        return torch.stack(class_losses).mean()
