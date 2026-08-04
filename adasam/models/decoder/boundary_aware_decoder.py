"""Lightweight decoder with auxiliary boundary supervision and gated refinement."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adasam.models.decoder.lightweight_decoder import LightweightSemanticDecoder


class BoundaryAwareSemanticDecoder(LightweightSemanticDecoder):
    """Add a small boundary branch without changing the baseline FPN topology."""

    def __init__(self, *args, enable_boundary_fusion: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        decoder_dim = self.classifier.in_channels
        boundary_dim = 32
        self.boundary_features = nn.Sequential(
            nn.Conv2d(decoder_dim, boundary_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, boundary_dim),
            nn.GELU(),
        )
        self.boundary_head = nn.Conv2d(boundary_dim, 1, 1)
        self.boundary_residual = nn.Conv2d(boundary_dim, decoder_dim, 1, bias=False)
        self.boundary_gate = nn.Conv2d(boundary_dim, 1, 1)
        self.enable_boundary_fusion = enable_boundary_fusion
        self.alpha = nn.Parameter(torch.zeros(()))
        self.last_boundary_logits: torch.Tensor | None = None

    def forward(
        self,
        features: dict[str, torch.Tensor],
        output_size: tuple[int, int],
        prompt_tokens: torch.Tensor | None = None,
        prompt: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        semantic = self.forward_features(features, prompt_tokens=prompt_tokens, prompt=prompt)
        boundary = self.boundary_features(semantic)
        boundary_logits = self.boundary_head(boundary)
        self.last_boundary_logits = F.interpolate(
            boundary_logits, size=output_size, mode="bilinear", align_corners=False
        )
        if self.enable_boundary_fusion:
            gate = torch.sigmoid(self.boundary_gate(boundary))
            semantic = semantic + self.alpha * gate * self.boundary_residual(boundary)
        logits = self.classifier(semantic)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
