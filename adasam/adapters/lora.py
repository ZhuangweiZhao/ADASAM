"""Minimal LoRA injection for linear layers in the TinyViT encoder."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Frozen linear projection plus a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        factory_kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, **factory_kwargs))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, **factory_kwargs))
        self.scale = float(alpha) / rank
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = F.linear(F.linear(inputs, self.lora_a), self.lora_b)
        return self.base(inputs) + residual * self.scale


def inject_tinyvit_lora(
    image_encoder: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    targets: tuple[str, ...] = ("qkv", "proj"),
) -> list[str]:
    """Replace TinyViT attention linear layers and return injected module names."""
    injected = []
    for name, module in list(image_encoder.named_modules()):
        if not isinstance(module, nn.Linear) or name.rsplit(".", 1)[-1] not in targets:
            continue
        if ".attn." not in f".{name}.":
            continue
        parent_name, child_name = name.rsplit(".", 1)
        parent = image_encoder.get_submodule(parent_name)
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha))
        injected.append(name)
    if not injected:
        raise RuntimeError("No TinyViT attention linear layers matched the LoRA targets")
    return injected
