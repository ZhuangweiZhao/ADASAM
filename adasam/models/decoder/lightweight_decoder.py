"""Lightweight FPN decoder for label-efficient semantic segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )


class LightweightSemanticDecoder(nn.Module):
    """Fuse P3, P4 and MobileSAM embedding features with a compact FPN."""

    def __init__(
        self,
        num_classes: int,
        feature_dims: dict[str, int] | None = None,
        decoder_dim: int = 96,
        enable_prompt_fusion: bool = False,
    ) -> None:
        super().__init__()
        dims = feature_dims or {"P3": 128, "P4": 160, "embedding": 256}
        self.lateral = nn.ModuleDict(
            {name: nn.Conv2d(channels, decoder_dim, 1) for name, channels in dims.items()}
        )
        self.p4_fuse = ConvNormAct(decoder_dim, decoder_dim)
        self.p3_fuse = ConvNormAct(decoder_dim, decoder_dim)
        self.refine = ConvNormAct(decoder_dim, decoder_dim)
        self.prompt_norm = nn.LayerNorm(256) if enable_prompt_fusion else None
        self.prompt_scale = nn.Linear(256, decoder_dim) if enable_prompt_fusion else None
        self.prompt_shift = nn.Linear(256, decoder_dim) if enable_prompt_fusion else None
        self.classifier = nn.Conv2d(decoder_dim, num_classes, 1)
        if self.prompt_scale is not None and self.prompt_shift is not None:
            nn.init.zeros_(self.prompt_scale.weight)
            nn.init.zeros_(self.prompt_scale.bias)
            nn.init.zeros_(self.prompt_shift.weight)
            nn.init.zeros_(self.prompt_shift.bias)

    def forward(
        self,
        features: dict[str, torch.Tensor],
        output_size: tuple[int, int],
        prompt_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        required = {"P3", "P4", "embedding"}
        missing = required - features.keys()
        if missing:
            raise KeyError(f"missing decoder features: {sorted(missing)}")
        p3 = self.lateral["P3"](features["P3"])
        p4 = self.lateral["P4"](features["P4"])
        embedding = self.lateral["embedding"](features["embedding"])
        embedding = F.interpolate(
            embedding, size=p4.shape[-2:], mode="bilinear", align_corners=False
        )
        p4 = self.p4_fuse(p4 + embedding)
        p4 = F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.refine(self.p3_fuse(p3 + p4))
        if prompt_tokens is not None:
            if self.prompt_norm is None or self.prompt_scale is None or self.prompt_shift is None:
                raise RuntimeError("decoder prompt fusion is disabled")
            if prompt_tokens.ndim != 3 or prompt_tokens.shape[0] != fused.shape[0]:
                raise ValueError("prompt_tokens must have shape [B,N,256]")
            prompt = self.prompt_norm(prompt_tokens.mean(dim=1))
            scale = torch.tanh(self.prompt_scale(prompt)).unsqueeze(-1).unsqueeze(-1)
            shift = self.prompt_shift(prompt).unsqueeze(-1).unsqueeze(-1)
            fused = fused * (1.0 + scale) + shift
        logits = self.classifier(fused)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
