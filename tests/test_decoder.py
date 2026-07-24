"""
SemanticMaskDecoder 单元测试 | Unit tests for SemanticMaskDecoder.
==================================================================

解码器测试需 weights/mobile_sam.pt (缺失则 skip)。
Decoder tests need weights/mobile_sam.pt (skip if absent).

接口 v4: sparse_token [1,C] (单 token) + dense_prompt [1,C,gh,gw].
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from adasam.decoder import SemanticMaskDecoder, SemanticMaskDecoderConfig

_CKPT = Path(__file__).resolve().parents[1] / "weights" / "mobile_sam.pt"
_skip_ckpt = pytest.mark.skipif(not _CKPT.exists(), reason="weights/mobile_sam.pt not present")


@pytest.fixture(scope="module")
def sam():
    from adasam.backbone import build_mobile_sam
    return build_mobile_sam(_CKPT, device="cpu")


@pytest.fixture(scope="module")
def decoder(sam):
    return SemanticMaskDecoder(sam.prompt_encoder, sam.mask_decoder, SemanticMaskDecoderConfig())


@_skip_ckpt
def test_single_token_decode_contract(decoder):
    """[1,C] token → ([1,1,256,256], [1,1]) | single-token decode shape contract."""
    torch.manual_seed(0)
    emb = torch.randn(1, 256, 64, 64)
    token = torch.randn(1, 256)  # single token, not N tokens
    low_res, iou = decoder(emb, token)
    assert low_res.shape == (1, 1, 256, 256)
    assert iou.shape == (1, 1)


@_skip_ckpt
def test_decode_with_dense_prompt(decoder):
    """dense_prompt [1,C,gh,gw] replaces no_mask_embed."""
    torch.manual_seed(0)
    emb = torch.randn(1, 256, 64, 64)
    token = torch.randn(1, 256)
    dense = torch.randn(1, 256, 64, 64)
    low_res_a, _ = decoder(emb, token)
    low_res_b, _ = decoder(emb, token, dense_prompt=dense)
    assert low_res_a.shape == (1, 1, 256, 256)
    assert low_res_b.shape == (1, 1, 256, 256)
    assert not torch.allclose(low_res_a, low_res_b)


@_skip_ckpt
def test_single_token_differentiable(decoder):
    """Gradients reach token and mask_decoder params."""
    emb = torch.randn(1, 256, 64, 64)
    token = torch.randn(1, 256, requires_grad=True)
    low_res, _ = decoder(emb, token)
    low_res.sum().backward()
    assert token.grad is not None and token.grad.abs().sum() > 0
    any_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in decoder.mask_decoder.parameters()
    )
    assert any_grad
    decoder.zero_grad(set_to_none=True)


@_skip_ckpt
def test_upscale_logits_shape(decoder):
    """[1,1,256,256] → [1,896,896] | upscale single mask."""
    low_res = torch.randn(1, 1, 256, 256)
    out = decoder.upscale_logits(low_res, input_size=(1024, 1024), original_size=(896, 896))
    assert out.shape == (1, 896, 896)
