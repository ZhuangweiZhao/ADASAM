"""Class-level semantic loss for prototype-conditioned semantic queries."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adasam.model.prototype_query_model import PrototypeQueryOutput


class PrototypeQuerySemanticLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        diversity_weight: float = 0.05,
        auxiliary_weight: float = 0.5,
        boundary_weight: float = 0.2,
        consistency_weight: float = 0.0,
        tversky_weight: float = 1.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        tversky_gamma: float = 1.33,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.diversity_weight = diversity_weight
        self.auxiliary_weight = auxiliary_weight
        self.boundary_weight = boundary_weight
        self.consistency_weight = consistency_weight
        self.tversky_weight = tversky_weight
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.tversky_gamma = tversky_gamma

    def _semantic_loss(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target = F.interpolate(
            target[:, None].float(), size=logits.shape[-2:], mode="nearest"
        )[:, 0]
        bce = F.binary_cross_entropy_with_logits(logits, target)
        gx = F.pad(torch.abs(target[:, 1:] - target[:, :-1]), (0, 0, 0, 1))
        gy = F.pad(torch.abs(target[:, :, 1:] - target[:, :, :-1]), (0, 1, 0, 0))
        boundary = (gx + gy).clamp_max(1.0)
        pred = logits.sigmoid()
        pred_boundary = F.pad(torch.abs(pred[:, 1:] - pred[:, :-1]), (0, 0, 0, 1)) + F.pad(torch.abs(pred[:, :, 1:] - pred[:, :, :-1]), (0, 1, 0, 0))
        boundary_loss = F.l1_loss(pred_boundary, boundary)
        probability = logits.sigmoid().flatten(1)
        target_flat = target.flatten(1)
        intersection = (probability * target_flat).sum(dim=1)
        denominator = probability.sum(dim=1) + target_flat.sum(dim=1)
        dice = (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        false_positive = (probability * (1.0 - target_flat)).sum(dim=1)
        false_negative = ((1.0 - probability) * target_flat).sum(dim=1)
        tversky = (intersection + 1.0) / (intersection + self.tversky_alpha * false_positive + self.tversky_beta * false_negative + 1.0)
        focal_tversky = (1.0 - tversky).pow(self.tversky_gamma).mean()
        return bce, dice, boundary_loss, focal_tversky

    @staticmethod
    def _diversity(output: PrototypeQueryOutput) -> torch.Tensor:
        components = output.query_mask_logits.sigmoid().flatten(2)
        components = F.normalize(components, dim=-1, eps=1e-6)
        similarity = components @ components.transpose(1, 2)
        q = similarity.shape[-1]
        if q < 2:
            return similarity.sum() * 0.0
        indices = torch.triu_indices(q, q, offset=1, device=similarity.device)
        relevance = output.query_logits.sigmoid()
        pair_weight = relevance[:, indices[0]] * relevance[:, indices[1]]
        return (similarity[:, indices[0], indices[1]] * pair_weight).mean()

    def forward(
        self,
        output: PrototypeQueryOutput,
        target: torch.Tensor,
        class_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        bce, dice, boundary, tversky = self._semantic_loss(output.semantic_logits, target)
        diversity = self._diversity(output)
        auxiliary = output.semantic_logits.sum() * 0.0
        for layer in output.auxiliary:
            layer_bce, layer_dice, layer_boundary, layer_tversky = self._semantic_loss(layer["semantic_logits"], target)
            auxiliary = auxiliary + self.bce_weight * layer_bce + self.dice_weight * layer_dice + self.boundary_weight * layer_boundary + self.tversky_weight * layer_tversky
        if output.auxiliary:
            auxiliary = auxiliary / len(output.auxiliary)
        loss = class_weight * (
            self.bce_weight * bce
            + self.dice_weight * dice
            + self.boundary_weight * boundary
            + self.tversky_weight * tversky
            + self.diversity_weight * diversity
            + self.auxiliary_weight * auxiliary
        )
        return {
            "loss": loss,
            "bce": bce,
            "dice": dice,
            "boundary": boundary,
            "tversky": tversky,
            "diversity": diversity,
            "auxiliary": auxiliary,
        }
