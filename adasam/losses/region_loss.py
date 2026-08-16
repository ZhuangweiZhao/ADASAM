"""Region-level multiclass losses aligned with semantic IoU."""

from __future__ import annotations

import torch
import torch.nn as nn


def _lovasz_gradient(sorted_foreground: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovasz extension of the Jaccard loss."""
    count = sorted_foreground.numel()
    total_positive = sorted_foreground.sum()
    intersection = total_positive - sorted_foreground.cumsum(0)
    union = total_positive + (1.0 - sorted_foreground).cumsum(0)
    gradient = 1.0 - intersection / union.clamp_min(1e-8)
    if count > 1:
        gradient[1:] = gradient[1:] - gradient[:-1]
    return gradient


class LovaszSoftmaxLoss(nn.Module):
    """Multiclass Lovasz-Softmax over classes present in the target batch."""

    def __init__(
        self,
        ignore_index: int | None = None,
        class_weights: torch.Tensor | list[float] | tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        weights = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        if weights is not None:
            if weights.ndim != 1 or bool((weights < 0).any()) or not bool((weights > 0).any()):
                raise ValueError("class_weights must be a non-negative 1D vector with a positive entry")
        self.register_buffer("class_weights", weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or target.ndim != 3:
            raise ValueError("expected logits [B,C,H,W] and target [B,H,W]")
        if logits.shape[0] != target.shape[0] or logits.shape[-2:] != target.shape[-2:]:
            raise ValueError("logits and target batch/spatial shapes must match")
        probability = logits.softmax(dim=1).permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
        labels = target.reshape(-1).long()
        if self.ignore_index is not None:
            valid = labels != self.ignore_index
            probability, labels = probability[valid], labels[valid]
        if labels.numel() == 0:
            return logits.sum() * 0.0
        if self.class_weights is not None and self.class_weights.numel() != logits.shape[1]:
            raise ValueError(
                f"expected {logits.shape[1]} class weights, got {self.class_weights.numel()}"
            )
        losses = []
        active_weights = []
        for class_id in range(logits.shape[1]):
            foreground = (labels == class_id).to(probability.dtype)
            if not bool(foreground.any()):
                continue
            errors = (foreground - probability[:, class_id]).abs()
            sorted_errors, permutation = torch.sort(errors, descending=True)
            sorted_foreground = foreground[permutation]
            losses.append(torch.dot(sorted_errors, _lovasz_gradient(sorted_foreground)))
            active_weights.append(
                probability.new_tensor(1.0)
                if self.class_weights is None
                else self.class_weights[class_id].to(probability)
            )
        if not losses:
            return logits.sum() * 0.0
        weights = torch.stack(active_weights)
        return torch.sum(torch.stack(losses) * weights) / weights.sum().clamp_min(1e-8)
