"""
QueryMaskDecoder 单元测试 | Unit tests for QueryMaskDecoder.
==============================================================

PrototypeAdapter 测试为纯合成 (无权重); 解码器测试需 weights/mobile_sam.pt (缺失则 skip)。
PrototypeAdapter tests are pure-synthetic; decoder tests need weights/mobile_sam.pt
(skip if absent).

覆盖 | Covers:
    - PrototypeAdapter: 零初始化 → 初始零 token | zero-init → zero token at start.
    - QueryMaskDecoder: 查询解码契约、可微性、分块=整批、upscale 形状、原型缺失报错。
      query-decode contract, differentiability, chunked == full, upscale shape,
      missing-prototype error.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from adasam.decoder import PrototypeAdapter, QueryMaskDecoder, QueryMaskDecoderConfig

_CKPT = Path(__file__).resolve().parents[1] / "weights" / "mobile_sam.pt"
_skip_ckpt = pytest.mark.skipif(not _CKPT.exists(), reason="weights/mobile_sam.pt not present")


# ═══════════════════════════════════════════════════════════════════
# PrototypeAdapter (no weights)
# ═══════════════════════════════════════════════════════════════════

def test_proto_adapter_zero_init():
    """零初始化: 初始输出全零 token | zero-init: initial output is all-zero token."""
    adapter = PrototypeAdapter(embed_dim=256, n_tokens=2)
    out = adapter(torch.randn(256))
    assert out.shape == (2, 256)
    assert torch.count_nonzero(out) == 0


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
    """[8,256] 查询 → ([8,1,256,256], [8,1]) | query-decode shape contract."""
    torch.manual_seed(0)
    emb = torch.randn(1, 256, 64, 64)
    queries = torch.randn(8, 256)
    proto = F.normalize(torch.randn(256), dim=0)
    low_res, iou = decoder(emb, queries, proto)
    assert low_res.shape == (8, 1, 256, 256)
    assert iou.shape == (8, 1)


@_skip_ckpt
def test_query_decode_differentiable(decoder):
    """梯度可达查询与 mask_decoder 参数 | grads reach queries and mask_decoder params."""
    emb = torch.randn(1, 256, 64, 64)
    queries = torch.randn(4, 256, requires_grad=True)
    proto = F.normalize(torch.randn(256), dim=0)
    low_res, _ = decoder(emb, queries, proto)
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
    """分块解码与整批解码一致 | chunked decoding matches full-batch decoding."""
    torch.manual_seed(1)
    emb = torch.randn(1, 256, 64, 64)
    queries = torch.randn(7, 256)
    proto = F.normalize(torch.randn(256), dim=0)

    full = QueryMaskDecoder(sam.prompt_encoder, sam.mask_decoder, QueryMaskDecoderConfig())
    chunked = QueryMaskDecoder(
        sam.prompt_encoder, sam.mask_decoder, QueryMaskDecoderConfig(decode_chunk_size=3)
    )
    chunked.load_state_dict(full.state_dict())
    with torch.no_grad():
        low_a, iou_a = full(emb, queries, proto)
        low_b, iou_b = chunked(emb, queries, proto)
    # 不同 batch 形状的 GEMM 舍入差异 ~1e-4 量级 | GEMM rounding differs by batch shape
    assert torch.allclose(low_a, low_b, atol=5e-4)
    assert torch.allclose(iou_a, iou_b, atol=5e-4)


@_skip_ckpt
def test_missing_prototype_raises(decoder):
    """use_proto_token=True 且未给原型 → ValueError | missing prototype raises."""
    emb = torch.randn(1, 256, 64, 64)
    queries = torch.randn(2, 256)
    with pytest.raises(ValueError):
        decoder(emb, queries, prototype=None)


@_skip_ckpt
def test_upscale_logits_shape(decoder):
    """[N,1,256,256] → [N,896,896] | upscale to tile resolution."""
    low_res = torch.randn(3, 1, 256, 256)
    out = decoder.upscale_logits(low_res, input_size=(1024, 1024), original_size=(896, 896))
    assert out.shape == (3, 896, 896)
