from __future__ import annotations

import torch
import torch.nn as nn

from adasam.adapters import LoRALinear, inject_tinyvit_lora


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(8, 24)
        self.proj = nn.Linear(8, 8)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()
        self.mlp = nn.Linear(8, 8)


def test_tinyvit_lora_is_zero_initialized_and_trainable() -> None:
    encoder = nn.ModuleList([Block(), Block()])
    inputs = torch.randn(2, 5, 8)
    baseline = encoder[0].attn.qkv(inputs).detach()
    names = inject_tinyvit_lora(encoder, rank=2, alpha=4)
    assert len(names) == 4
    assert isinstance(encoder[0].attn.qkv, LoRALinear)
    assert torch.allclose(encoder[0].attn.qkv(inputs), baseline)
    encoder[0].attn.qkv(inputs).mean().backward()
    assert encoder[0].attn.qkv.lora_b.grad is not None
    assert encoder[0].attn.qkv.base.weight.grad is None
    assert encoder[0].mlp.weight.requires_grad


def test_lora_parameters_follow_module_device() -> None:
    encoder = nn.ModuleList([Block()]).to(dtype=torch.float64)
    inject_tinyvit_lora(encoder, rank=2, alpha=4)
    encoder.to(torch.device("cpu"))
    assert encoder[0].attn.qkv.lora_a.device == encoder[0].attn.qkv.base.weight.device
    assert encoder[0].attn.qkv.lora_b.device == encoder[0].attn.qkv.base.weight.device
    assert encoder[0].attn.qkv.lora_a.dtype == encoder[0].attn.qkv.base.weight.dtype
    assert encoder[0].attn.qkv.lora_b.dtype == encoder[0].attn.qkv.base.weight.dtype
