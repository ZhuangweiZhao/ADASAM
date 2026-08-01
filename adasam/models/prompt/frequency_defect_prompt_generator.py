"""Frequency-spatial defect prompts for label-efficient segmentation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FrequencyBlock(nn.Module):
    """Extract a normalized log-magnitude spectrum from an adapted feature map."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_dtype = feature.dtype
        # CUDA FFT does not support every low-precision/shape combination reliably.
        spectrum = torch.fft.fft2(feature.float(), dim=(-2, -1), norm="ortho")
        magnitude = torch.log1p(spectrum.abs())
        mean = magnitude.mean(dim=(-2, -1), keepdim=True)
        variance = magnitude.var(dim=(-2, -1), keepdim=True, unbiased=False)
        normalized = (magnitude - mean) * torch.rsqrt(variance + 1e-6)
        normalized = normalized.to(input_dtype)
        return self.refine(normalized), magnitude.to(input_dtype)


class FrequencyAwareDefectPromptGenerator(nn.Module):
    """Generate dense and token prompts from spatial-frequency defect features."""

    def __init__(
        self,
        feature_dims: dict[str, int] | None = None,
        prompt_dim: int = 256,
        num_prompt: int = 8,
        fusion_dim: int = 32,
        attention_dim: int = 64,
        num_heads: int = 4,
        dense_size: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        if attention_dim % num_heads:
            raise ValueError("attention_dim must be divisible by num_heads")
        dims = feature_dims or {"P3": 128, "P4": 160, "embedding": 256}
        required = {"P3", "P4", "embedding"}
        if missing := required - dims.keys():
            raise KeyError(f"missing feature dimensions: {sorted(missing)}")

        self.num_prompt = num_prompt
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.dense_size = dense_size
        self.spatial_lateral = nn.ModuleDict(
            {name: nn.Conv2d(dims[name], fusion_dim, 1, bias=False) for name in required}
        )
        self.frequency_lateral = nn.ModuleDict(
            {name: nn.Conv2d(dims[name], fusion_dim, 1, bias=False) for name in ("P3", "P4")}
        )
        self.frequency_blocks = nn.ModuleDict(
            {name: _FrequencyBlock(fusion_dim) for name in ("P3", "P4")}
        )
        self.frequency_projection = nn.Conv2d(fusion_dim, fusion_dim, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.dense_head = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, fusion_dim),
            nn.GELU(),
            nn.Conv2d(fusion_dim, prompt_dim, 1, bias=False),
        )

        self.learnable_tokens = nn.Parameter(torch.empty(num_prompt, prompt_dim))
        self.query_proj = nn.Linear(prompt_dim, attention_dim, bias=False)
        self.key_proj = nn.Conv2d(fusion_dim, attention_dim, 1, bias=False)
        self.value_proj = nn.Conv2d(fusion_dim, attention_dim, 1, bias=False)
        self.output_proj = nn.Linear(attention_dim, prompt_dim, bias=False)
        self.token_norm = nn.LayerNorm(prompt_dim)
        nn.init.normal_(self.learnable_tokens, std=0.02)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        required = {"P3", "P4", "embedding"}
        if missing := required - features.keys():
            raise KeyError(f"FDAPG-v3 requires features: {sorted(missing)}")
        batch = features["embedding"].shape[0]

        spatial_parts = []
        for name in ("P3", "P4", "embedding"):
            projected = self.spatial_lateral[name](features[name])
            spatial_parts.append(
                F.interpolate(projected, self.dense_size, mode="bilinear", align_corners=False)
            )
        spatial = sum(spatial_parts)

        frequency_parts = []
        magnitude_parts = []
        for name in ("P3", "P4"):
            projected = self.frequency_lateral[name](features[name])
            frequency, magnitude = self.frequency_blocks[name](projected)
            frequency_parts.append(
                F.interpolate(frequency, self.dense_size, mode="bilinear", align_corners=False)
            )
            magnitude_parts.append(
                F.interpolate(
                    magnitude.mean(dim=1, keepdim=True),
                    self.dense_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        frequency = self.frequency_projection(sum(frequency_parts) / len(frequency_parts))
        fused = spatial + self.alpha * frequency
        dense_prompt = self.dense_head(fused)

        query = self.query_proj(self.learnable_tokens).expand(batch, -1, -1)
        key = self.key_proj(fused).flatten(2).transpose(1, 2)
        value = self.value_proj(fused).flatten(2).transpose(1, 2)
        tokens = self.dense_size[0] * self.dense_size[1]
        query = query.view(batch, self.num_prompt, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        attention = torch.softmax(query @ key.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1)
        context = (attention @ value).transpose(1, 2).reshape(batch, self.num_prompt, -1)
        token_prompt = self.token_norm(
            self.learnable_tokens.unsqueeze(0) + self.output_proj(context)
        )

        return {
            "dense_prompt": dense_prompt,
            "token_prompt": token_prompt,
            "frequency_heatmap": sum(magnitude_parts) / len(magnitude_parts),
            "dense_activation": dense_prompt.abs().mean(dim=1, keepdim=True),
        }

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
