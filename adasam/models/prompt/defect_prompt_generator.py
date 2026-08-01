"""Defect-aware domain prompts generated from the current image features."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class DefectPromptGenerator(nn.Module):
    """Generate image-conditioned defect tokens with low-rank cross-attention.

    Attention is performed in a compact internal space so the prompt generator stays
    below 100K parameters while returning decoder-compatible 256-D tokens.
    """

    def __init__(
        self,
        p4_dim: int = 160,
        prompt_dim: int = 256,
        num_prompt: int = 16,
        attention_dim: int = 64,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if attention_dim % num_heads:
            raise ValueError("attention_dim must be divisible by num_heads")
        self.num_prompt = num_prompt
        self.prompt_dim = prompt_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.learnable_tokens = nn.Parameter(torch.empty(num_prompt, prompt_dim))
        self.condition = nn.Sequential(nn.Linear(p4_dim, prompt_dim), nn.GELU())
        self.query_proj = nn.Linear(prompt_dim, attention_dim, bias=False)
        self.key_proj = nn.Conv2d(p4_dim, attention_dim, 1, bias=False)
        self.value_proj = nn.Conv2d(p4_dim, attention_dim, 1, bias=False)
        self.output_proj = nn.Linear(attention_dim, prompt_dim, bias=False)
        self.norm = nn.LayerNorm(prompt_dim)
        nn.init.normal_(self.learnable_tokens, std=0.02)
        nn.init.zeros_(self.output_proj.weight)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        if "P4" not in features:
            raise KeyError("DAPG requires the P4 feature")
        p4 = features["P4"]
        batch, _, height, width = p4.shape
        pooled = p4.mean(dim=(2, 3))
        conditioned = self.learnable_tokens.unsqueeze(0) + self.condition(pooled).unsqueeze(1)
        query = self.query_proj(conditioned)
        key = self.key_proj(p4).flatten(2).transpose(1, 2)
        value = self.value_proj(p4).flatten(2).transpose(1, 2)

        query = query.view(batch, self.num_prompt, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, height * width, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, height * width, self.num_heads, self.head_dim).transpose(1, 2)
        attention = torch.softmax((query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        context = (attention @ value).transpose(1, 2).reshape(batch, self.num_prompt, -1)
        return self.norm(conditioned + self.output_proj(context))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
