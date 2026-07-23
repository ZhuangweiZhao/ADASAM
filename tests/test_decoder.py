"""
QueryMaskDecoder 单元测试 | Unit tests for QueryMaskDecoder.
==============================================================

解码器测试需 weights/mobile_sam.pt (缺失则 skip)。
Decoder tests need weights/mobile_sam.pt (skip if absent).

覆盖 | Covers:
    - QueryMaskDecoder: query-decode contract, differentiability, chunked==full,
      upscale shape, dense_prompt_override.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from adasam.decoder import QueryMaskDecoder, QueryMaskDecoderConfig

_CKPT = Path(__file__).resolve().parents[1] / "weights" / "mobile_sam.pt"
_skip_ckpt = pytest.mark.skipif(not _CKPT.exists(), reason="weights/mobile_sam.pt not present")


# ═══════════════════════════════════════════════════════════════════
# QueryMaskDecoder (weight-dependent)
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def sam():
    from adasam.backbone import build_mobile_sam
    return build_mobile_sam(_CKPT, device="cpu")


@pytest.fixture(scope="module")
def decoder(sam):
    return QueryMaskDecoder(sam.prompt_encoder, sam.mask_decoder, QueryMaskDecoderConfig())


@_skip_ckpt
def test_query_decode_contract(decoder):
    """[8,256] queries → ([8,1,256,256], [8,1]) | query-decode shape contract."""
    torch.manual_seed(0)
    emb = torch.randn(1, 256, 64, 64)
    queries = torch.randn(8, 256)
    low_res, iou = decoder(emb, queries)
    assert low_res.shape == (8, 1, 256, 256)
    assert iou.shape == (8, 1)


@_skip_ckpt
def test_query_decode_with_dense_override(decoder):
    """dense_prompt_override [1,C,gh,gw] replaces no_mask_embed."""
    torch.manual_seed(0)
    emb = torch.randn(1, 256, 64, 64)
    queries = torch.randn(8, 256)
    dense = torch.randn(1, 256, 64, 64)
    low_res_a, _ = decoder(emb, queries)                       # default: no_mask_embed
    low_res_b, _ = decoder(emb, queries, dense_prompt_override=dense)
    # Different dense prompts ⇒ different output
    assert low_res_a.shape == (8, 1, 256, 256)
    assert low_res_b.shape == (8, 1, 256, 256)
    assert not torch.allclose(low_res_a, low_res_b)


@_skip_ckpt
def test_query_decode_differentiable(decoder):
    """Gradients reach queries and mask_decoder params."""
    emb = torch.randn(1, 256, 64, 64)
    queries = torch.randn(4, 256, requires_grad=True)
    low_res, _ = decoder(emb, queries)
    low_res.sum().backward()
    assert queries.grad is not None and queries.grad.abs().sum() > 0
    any_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in decoder.mask_decoder.parameters()
    )
    assert any_grad
    decoder.zero_grad(set_to_none=True)


@_skip_ckpt
def test_query_decode_chunked_equals_full(sam):
    """Chunked decoding matches full-batch decoding."""
    torch.manual_seed(1)
    emb = torch.randn(1, 256, 64, 64)
    queries = torch.randn(7, 256)

    full = QueryMaskDecoder(sam.prompt_encoder, sam.mask_decoder, QueryMaskDecoderConfig())
    chunked = QueryMaskDecoder(
        sam.prompt_encoder, sam.mask_decoder, QueryMaskDecoderConfig(decode_chunk_size=3)
    )
    chunked.load_state_dict(full.state_dict())
    with torch.no_grad():
        low_a, iou_a = full(emb, queries)
        low_b, iou_b = chunked(emb, queries)
    # GEMM rounding differs by batch shape ~1e-4
    assert torch.allclose(low_a, low_b, atol=5e-4)
    assert torch.allclose(iou_a, iou_b, atol=5e-4)


@_skip_ckpt
def test_upscale_logits_shape(decoder):
    """[N,1,256,256] → [N,896,896] | upscale to tile resolution."""
    low_res = torch.randn(3, 1, 256, 256)
    out = decoder.upscale_logits(low_res, input_size=(1024, 1024), original_size=(896, 896))
    assert out.shape == (3, 896, 896)
