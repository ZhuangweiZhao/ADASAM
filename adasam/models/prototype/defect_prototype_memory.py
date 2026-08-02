"""Class-aware prototype memory for industrial defect segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DefectPrototypeMemory(nn.Module):
    """Maintain EMA class prototypes and inject their similarity as a feature prior."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 256,
        momentum: float = 0.9,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must include background and at least one defect class")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.prior_projection = nn.Conv2d(num_classes, feature_dim, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.register_buffer("prototypes", torch.zeros(num_classes, feature_dim))
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer("update_counts", torch.zeros(num_classes, dtype=torch.long))

    def _resized_target(self, target: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if target.ndim != 3:
            raise ValueError("target must have shape [B,H,W]")
        return F.interpolate(target.unsqueeze(1).float(), size=size, mode="nearest").squeeze(1).long()

    @torch.no_grad()
    def update(self, feature: torch.Tensor, target: torch.Tensor) -> None:
        """Update every class present in the batch using mask-pooled detached features."""
        target_small = self._resized_target(target, feature.shape[-2:])
        for class_id in range(self.num_classes):
            mask = target_small == class_id
            if not mask.any():
                continue
            values = feature.permute(0, 2, 3, 1)[mask]
            pooled = values.mean(dim=0)
            if self.initialized[class_id]:
                self.prototypes[class_id].mul_(self.momentum).add_(
                    pooled, alpha=1.0 - self.momentum
                )
            else:
                self.prototypes[class_id].copy_(pooled)
                self.initialized[class_id] = True
            self.update_counts[class_id] += 1

    def forward(
        self,
        feature: torch.Tensor,
        target: torch.Tensor | None = None,
        update_memory: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if feature.ndim != 4 or feature.shape[1] != self.feature_dim:
            raise ValueError(
                f"feature must have shape [B,{self.feature_dim},H,W], got {tuple(feature.shape)}"
            )
        if update_memory is None:
            update_memory = self.training and target is not None
        if update_memory:
            if target is None:
                raise ValueError("target is required when update_memory=True")
            self.update(feature.detach(), target)

        normalized_feature = F.normalize(feature, dim=1, eps=1e-6)
        normalized_prototypes = F.normalize(self.prototypes, dim=1, eps=1e-6)
        similarity = torch.einsum("bdhw,cd->bchw", normalized_feature, normalized_prototypes)
        similarity = similarity * self.initialized.to(similarity.dtype).view(1, -1, 1, 1)
        prototype_response = self.prior_projection(similarity)
        enhanced = feature + self.alpha * prototype_response
        return enhanced, {
            "source_feature": feature,
            "similarity": similarity,
            "prototype_response": prototype_response,
            "enhanced_feature": enhanced,
            "prototypes": self.prototypes.detach().clone(),
            "initialized": self.initialized.detach().clone(),
        }
