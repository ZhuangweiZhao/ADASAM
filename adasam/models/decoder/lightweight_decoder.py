"""Lightweight FPN decoder for label-efficient semantic segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adasam.adapters import CATAdapter


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
        feature_scales: str = "p3_p4_embedding",
        post_fusion_adapter: bool = False,
        adapter_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        if spatial_prompt_mode not in {"both", "dense", "token"}:
            raise ValueError("spatial_prompt_mode must be one of: both, dense, token")
        scale_names = {
            "embedding": ("embedding",),
            "p4_embedding": ("P4", "embedding"),
            "p3_p4_embedding": ("P3", "P4", "embedding"),
        }
        if feature_scales not in scale_names:
            raise ValueError(
                "feature_scales must be one of: embedding, p4_embedding, p3_p4_embedding"
            )
        self.feature_scales = feature_scales
        self.feature_names = scale_names[feature_scales]
        dims = feature_dims or {"P3": 128, "P4": 160, "embedding": 256}
        self.lateral = nn.ModuleDict(
            {
                name: nn.Conv2d(dims[name], decoder_dim, 1)
                for name in self.feature_names
            }
        )
        self.p4_fuse = (
            ConvNormAct(decoder_dim, decoder_dim) if "P4" in self.feature_names else None
        )
        self.p3_fuse = (
            ConvNormAct(decoder_dim, decoder_dim) if "P3" in self.feature_names else None
        )
        self.post_fusion_adapter = (
            CATAdapter(
                dim=decoder_dim,
                bottleneck=max(8, int(round(decoder_dim * adapter_ratio))),
            )
            if post_fusion_adapter
            else None
        )
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

    def forward_features(
        self,
        features: dict[str, torch.Tensor],
        prompt_tokens: torch.Tensor | None = None,
        prompt: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        required = set(self.feature_names)
        missing = required - features.keys()
        if missing:
            raise KeyError(f"missing decoder features: {sorted(missing)}")
        fused = self.lateral["embedding"](features["embedding"])
        if "P4" in self.feature_names:
            p4 = self.lateral["P4"](features["P4"])
            fused = F.interpolate(fused, size=p4.shape[-2:], mode="bilinear", align_corners=False)
            fused = self.p4_fuse(p4 + fused)
        if "P3" in self.feature_names:
            p3 = self.lateral["P3"](features["P3"])
            fused = F.interpolate(fused, size=p3.shape[-2:], mode="bilinear", align_corners=False)
            fused = self.p3_fuse(p3 + fused)
        if self.post_fusion_adapter is not None:
            fused = self.post_fusion_adapter(fused)
        fused = self.refine(fused)
        if prompt_tokens is not None:
            if self.prompt_norm is None or self.prompt_scale is None or self.prompt_shift is None:
                raise RuntimeError("decoder prompt fusion is disabled")
            if prompt_tokens.ndim != 3 or prompt_tokens.shape[0] != fused.shape[0]:
                raise ValueError("prompt_tokens must have shape [B,N,256]")
            token_summary = self.prompt_norm(prompt_tokens.mean(dim=1))
            scale = torch.tanh(self.prompt_scale(token_summary)).unsqueeze(-1).unsqueeze(-1)
            shift = self.prompt_shift(token_summary).unsqueeze(-1).unsqueeze(-1)
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
        return fused

    def forward(
        self,
        features: dict[str, torch.Tensor],
        output_size: tuple[int, int],
        prompt_tokens: torch.Tensor | None = None,
        prompt: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        fused = self.forward_features(features, prompt_tokens=prompt_tokens, prompt=prompt)
        logits = self.classifier(fused)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
