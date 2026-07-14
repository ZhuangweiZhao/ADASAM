"""
Matcher 与 PromptMaskDecoder 单元测试 | Unit tests for Matcher & PromptMaskDecoder.
====================================================================================

Matcher 测试为纯合成 (无权重); 解码器测试需 weights/mobile_sam.pt (缺失则 skip)。
Matcher tests are pure-synthetic; decoder tests need weights/mobile_sam.pt (skip if absent).

覆盖 | Covers:
    - similarity_map: 形状、余弦范围 | shape, cosine range.
    - Matcher: ≥1 点兜底、top_k 上限、阈值、坐标落在输入帧、NMS 分离多峰。
    - PrototypeAdapter: 零初始化 → 初始零 token | zero-init → zero token at start.
    - PromptMaskDecoder: forward 契约、多峰→多实例、decode 低分辨率形状、scale_points。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from adasam.prototype import Matcher, similarity_map
from adasam.decoder import PromptMaskDecoder, PrototypeAdapter

_CKPT = Path(__file__).resolve().parents[1] / "weights" / "mobile_sam.pt"
_skip_ckpt = pytest.mark.skipif(not _CKPT.exists(), reason="weights/mobile_sam.pt not present")


# ═══════════════════════════════════════════════════════════════════
# similarity_map + Matcher (no weights)
# ═══════════════════════════════════════════════════════════════════

def test_similarity_map_shape_and_range():
    emb = torch.randn(256, 64, 64)
    proto = F.normalize(torch.randn(256), dim=0)
    sim = similarity_map(emb, proto)
    assert sim.shape == (64, 64)
    assert float(sim.min()) >= -1.0001 and float(sim.max()) <= 1.0001


def test_matcher_always_returns_at_least_one():
    """全低相似度 → 兜底返回全局最大 1 点 | all-low sim → global-max fallback (1 point)."""
    sim = torch.full((64, 64), -0.9)
    sim[10, 20] = -0.1                                    # 仍低于阈值 | still below threshold
    pts = Matcher(top_k=10, sim_threshold=0.5).select(sim, stride=16.0)
    assert pts.coords.shape == (1, 2)
    # 兜底点应为全局最大处 (x=(20+.5)*16, y=(10+.5)*16) | fallback is the global max
    assert pts.coords[0, 0].item() == pytest.approx((20 + 0.5) * 16)
    assert pts.coords[0, 1].item() == pytest.approx((10 + 0.5) * 16)


def test_matcher_topk_and_separation():
    """三个分离峰 → 恰好 3 点, 坐标在输入帧内 | three separated peaks → 3 points in input frame."""
    sim = torch.zeros(64, 64)
    for (gy, gx) in [(5, 5), (30, 30), (55, 55)]:
        sim[gy, gx] = 1.0
    pts = Matcher(top_k=10, sim_threshold=0.5, min_distance=1).select(sim, stride=16.0)
    assert pts.coords.shape[0] == 3
    assert pts.labels.tolist() == [1.0, 1.0, 1.0]
    assert float(pts.coords.min()) >= 0.0 and float(pts.coords.max()) <= 1024.0


def test_matcher_respects_top_k_cap():
    """峰数多于 top_k 时截断 | more peaks than top_k → capped."""
    sim = torch.zeros(64, 64)
    sim[::4, ::4] = 1.0                                   # 大量峰 | many peaks
    pts = Matcher(top_k=5, sim_threshold=0.5, min_distance=1).select(sim, stride=16.0)
    assert pts.coords.shape[0] == 5


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
# PromptMaskDecoder (weight-dependent)
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def decoder():
    from adasam.backbone import build_mobile_sam
    sam = build_mobile_sam(_CKPT, device="cpu")
    return PromptMaskDecoder(sam.prompt_encoder, sam.mask_decoder, top_k_points=10, sim_threshold=0.5)


@_skip_ckpt
def test_decoder_forward_contract(decoder):
    """forward → (masks[N,896,896] bool, scores[N]∈[0,1]), N≥1 | inference contract."""
    emb = torch.randn(1, 256, 64, 64)
    proto = F.normalize(torch.randn(256), dim=0)
    out = decoder(emb, proto)
    n = out.masks.shape[0]
    assert n >= 1
    assert out.masks.shape == (n, 896, 896) and out.masks.dtype == torch.bool
    assert out.scores.shape == (n,)
    assert float(out.scores.min()) >= 0.0 and float(out.scores.max()) <= 1.0


@_skip_ckpt
def test_decoder_multi_peak_multi_instance(decoder):
    """三个匹配区域 → 三个实例 | three matching regions → three instances."""
    proto = torch.zeros(256); proto[0] = 1.0             # 原型 = e0 | prototype = e0
    emb = torch.zeros(1, 256, 64, 64)
    for (gy, gx) in [(8, 8), (32, 32), (56, 56)]:
        emb[0, 0, gy, gx] = 10.0                         # 这些格点归一化后≈e0 | ≈ e0 after norm
    out = decoder(emb, proto)
    assert out.masks.shape[0] == 3


@_skip_ckpt
def test_decoder_decode_low_res_shape(decoder):
    """decode() 返回低分辨率 logits [N,1,256,256] 与 iou [N,1] | decode() shapes."""
    emb = torch.randn(1, 256, 64, 64)
    proto = F.normalize(torch.randn(256), dim=0)
    coords = torch.tensor([[512.0, 512.0], [256.0, 256.0]])
    labels = torch.ones(2)
    low_res, iou = decoder.decode(emb, proto, coords, labels)
    assert low_res.shape == (2, 1, 256, 256)
    assert iou.shape == (2, 1)


@_skip_ckpt
def test_decoder_decode_is_differentiable(decoder):
    """decode() 对可训练参数可导 | decode() is differentiable w.r.t. trainable params."""
    emb = torch.randn(1, 256, 64, 64)
    proto = F.normalize(torch.randn(256), dim=0)
    coords = torch.tensor([[512.0, 512.0]])
    labels = torch.ones(1)
    low_res, _ = decoder.decode(emb, proto, coords, labels)
    low_res.sum().backward()
    # adapter 末层零初始化但梯度应非 None | adapter grad exists (even if zero-init)
    assert decoder.proto_adapter.fc1.weight.grad is not None


def test_scale_points_frame_mapping():
    """tile(896) → 输入(1024) 帧坐标线性缩放 | linear frame scaling 896→1024."""
    coords = torch.tensor([[448.0, 448.0]])              # 896 帧中心 | center in 896 frame
    scaled = PromptMaskDecoder.scale_points(coords, (896, 896), (1024, 1024))
    assert scaled[0, 0].item() == pytest.approx(448.0 * 1024 / 896)
    assert scaled[0, 1].item() == pytest.approx(448.0 * 1024 / 896)
