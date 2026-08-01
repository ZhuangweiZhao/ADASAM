"""Spatial defect-aware prompts for label-efficient industrial segmentation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DefectAwarePromptGeneratorV2(nn.Module):
    """Generate dense spatial priors and image-conditioned defect tokens.

    The dense branch retains the spatial layout of the multi-scale features, while
    the token branch uses learnable defect queries to attend over MobileSAM's image
    embedding.  Low-rank attention keeps the module lightweight.
    """

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
        self.prompt_dim = prompt_dim
        self.num_prompt = num_prompt
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.dense_size = dense_size

        self.dense_lateral = nn.ModuleDict(
            {name: nn.Conv2d(dims[name], fusion_dim, 1, bias=False) for name in required}
        )
        self.dense_fuse = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, fusion_dim),
            nn.GELU(),
            nn.Conv2d(fusion_dim, prompt_dim, 1, bias=False),
        )

        self.learnable_tokens = nn.Parameter(torch.empty(num_prompt, prompt_dim))
        self.query_proj = nn.Linear(prompt_dim, attention_dim, bias=False)
        self.key_proj = nn.Conv2d(dims["embedding"], attention_dim, 1, bias=False)
        self.value_proj = nn.Conv2d(dims["embedding"], attention_dim, 1, bias=False)
        self.output_proj = nn.Linear(attention_dim, prompt_dim, bias=False)
        self.token_norm = nn.LayerNorm(prompt_dim)
        nn.init.normal_(self.learnable_tokens, std=0.02)
        # Starts as a no-op in the decoder but still allows prompt learning.
        nn.init.zeros_(self.dense_fuse[-1].weight)
        nn.init.zeros_(self.output_proj.weight)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        required = {"P3", "P4", "embedding"}
        if missing := required - features.keys():
            raise KeyError(f"DAPG-v2 requires features: {sorted(missing)}")
        batch = features["embedding"].shape[0]
        dense_features = []
        for name in ("P3", "P4", "embedding"):
            feature = self.dense_lateral[name](features[name])
            dense_features.append(
                F.interpolate(feature, self.dense_size, mode="bilinear", align_corners=False)
            )
        dense_prompt = self.dense_fuse(sum(dense_features))

        embedding = features["embedding"]
        _, _, height, width = embedding.shape
        query = self.query_proj(self.learnable_tokens).expand(batch, -1, -1)
        key = self.key_proj(embedding).flatten(2).transpose(1, 2)
        value = self.value_proj(embedding).flatten(2).transpose(1, 2)
        query = query.view(batch, self.num_prompt, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, height * width, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, height * width, self.num_heads, self.head_dim).transpose(1, 2)
        attention = torch.softmax(query @ key.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1)
        context = (attention @ value).transpose(1, 2).reshape(batch, self.num_prompt, -1)
        token_prompt = self.token_norm(
            self.learnable_tokens.unsqueeze(0) + self.output_proj(context)
        )
        return {"dense_prompt": dense_prompt, "token_prompt": token_prompt}

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
