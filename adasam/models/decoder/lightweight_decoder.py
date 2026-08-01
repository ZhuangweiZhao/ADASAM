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
        enable_spatial_prompt_fusion: bool = False,
        spatial_prompt_mode: str = "both",
    ) -> None:
        super().__init__()
        if spatial_prompt_mode not in {"both", "dense", "token"}:
            raise ValueError("spatial_prompt_mode must be one of: both, dense, token")
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
        self.dense_prompt_proj = (
            nn.Conv2d(256, decoder_dim, 1, bias=False) if enable_spatial_prompt_fusion else None
        )
        self.token_query = (
            nn.Linear(decoder_dim, 64, bias=False) if enable_spatial_prompt_fusion else None
        )
        self.token_key = nn.Linear(256, 64, bias=False) if enable_spatial_prompt_fusion else None
        self.token_value = nn.Linear(256, 64, bias=False) if enable_spatial_prompt_fusion else None
        self.token_output = (
            nn.Linear(64, decoder_dim, bias=False) if enable_spatial_prompt_fusion else None
        )
        self.spatial_prompt_mode = spatial_prompt_mode
        self.classifier = nn.Conv2d(decoder_dim, num_classes, 1)
        if self.prompt_scale is not None and self.prompt_shift is not None:
            nn.init.zeros_(self.prompt_scale.weight)
            nn.init.zeros_(self.prompt_scale.bias)
            nn.init.zeros_(self.prompt_shift.weight)
            nn.init.zeros_(self.prompt_shift.bias)
        if self.dense_prompt_proj is not None:
            nn.init.zeros_(self.dense_prompt_proj.weight)
            nn.init.zeros_(self.token_output.weight)

    def forward(
        self,
        features: dict[str, torch.Tensor],
        output_size: tuple[int, int],
        prompt_tokens: torch.Tensor | None = None,
        prompt: dict[str, torch.Tensor] | None = None,
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
        if prompt is not None:
            if any(module is None for module in (self.dense_prompt_proj, self.token_query, self.token_key, self.token_value, self.token_output)):
                raise RuntimeError("decoder spatial prompt fusion is disabled")
            dense_prompt = prompt.get("dense_prompt")
            token_prompt = prompt.get("token_prompt")
            if self.spatial_prompt_mode in {"both", "dense"}:
                if dense_prompt is None:
                    raise KeyError("spatial prompt requires dense_prompt")
                if dense_prompt.ndim != 4 or dense_prompt.shape[:2] != (fused.shape[0], 256):
                    raise ValueError("dense_prompt must have shape [B,256,H,W]")
                dense = F.interpolate(dense_prompt, fused.shape[-2:], mode="bilinear", align_corners=False)
                fused = fused + self.dense_prompt_proj(dense)
            if self.spatial_prompt_mode in {"both", "token"}:
                if token_prompt is None:
                    raise KeyError("spatial prompt requires token_prompt")
                if token_prompt.ndim != 3 or token_prompt.shape[0] != fused.shape[0] or token_prompt.shape[2] != 256:
                    raise ValueError("token_prompt must have shape [B,N,256]")
                query = self.token_query(fused.flatten(2).transpose(1, 2))
                key = self.token_key(token_prompt)
                value = self.token_value(token_prompt)
                attention = torch.softmax(query @ key.transpose(-2, -1) / 8.0, dim=-1)
                modulation = self.token_output(attention @ value).transpose(1, 2).reshape_as(fused)
                fused = fused + modulation
        logits = self.classifier(fused)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
