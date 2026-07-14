"""
原型构建与记忆单元测试 | Unit tests for PrototypeBuilder & PrototypeMemory.
=============================================================================

覆盖 | Covers:
    - build: 形状 [256]、L2 归一化、空掩码→零、多 support 平均、判别性。
    - memory: add/get/has/classes、running mean、归一化、缺失报错。
纯合成张量, 无需真实数据/权重 | Pure synthetic tensors, no real data/weights needed.
"""

from __future__ import annotations

import pytest
import torch

from adasam.prototype import PrototypeBuilder, PrototypeMemory


# ═══════════════════════════════════════════════════════════════════
# PrototypeBuilder
# ═══════════════════════════════════════════════════════════════════

def test_build_shape_and_l2_norm():
    """原型形状 [256] 且 L2 范数≈1 | prototype shape [256], L2 norm ≈ 1."""
    builder = PrototypeBuilder(embed_dim=256)
    emb = torch.randn(256, 64, 64)
    mask = torch.zeros(896, 896)
    mask[100:400, 100:400] = 1
    proto = builder.build([emb], [mask])
    assert proto.shape == (256,)
    assert proto.norm().item() == pytest.approx(1.0, abs=1e-5)


def test_build_empty_mask_returns_zero():
    """全空掩码 → 零向量 | all-empty masks → zero vector."""
    builder = PrototypeBuilder(embed_dim=256)
    emb = torch.randn(256, 64, 64)
    proto = builder.build([emb], [torch.zeros(896, 896)])
    assert proto.shape == (256,)
    assert proto.norm().item() == pytest.approx(0.0, abs=1e-6)


def test_build_averages_multiple_supports():
    """多 support 求平均: 与单 support 结果一致时说明平均生效 | averaging over supports works."""
    builder = PrototypeBuilder(embed_dim=256)
    full = torch.ones(896, 896)
    emb_a = torch.randn(256, 64, 64)
    emb_b = torch.randn(256, 64, 64)
    p_both = builder.build([emb_a, emb_b], [full, full])
    assert p_both.shape == (256,) and p_both.norm().item() == pytest.approx(1.0, abs=1e-5)
    # 单一 support 应不同于两 support 平均 | single-support differs from the 2-support mean
    p_a = builder.build([emb_a], [full])
    assert not torch.allclose(p_a, p_both)


def test_build_discriminative_direction():
    """原型方向应贴近前景区域嵌入均值 | prototype aligns with FG-region embedding mean."""
    builder = PrototypeBuilder(embed_dim=8)
    emb = torch.zeros(8, 64, 64)
    # 在前景 64² 网格左上角放一个已知方向 | put a known direction in the top-left grid cell
    emb[:, :8, :8] = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0]).view(8, 1, 1)
    mask = torch.zeros(896, 896)
    mask[: 896 // 8, : 896 // 8] = 1                     # 映射到左上角网格 | maps to top-left cell
    proto = builder.build([emb], [mask])
    # 应几乎与 e0 同向 | should be ≈ e0
    assert proto[0].item() == pytest.approx(1.0, abs=1e-4)
    assert proto[1:].abs().max().item() == pytest.approx(0.0, abs=1e-4)


def test_build_length_mismatch_raises():
    """embeddings 与 masks 数量不等 → 报错 | mismatched lengths raise."""
    builder = PrototypeBuilder()
    with pytest.raises(ValueError):
        builder.build([torch.randn(256, 64, 64)], [])


# ═══════════════════════════════════════════════════════════════════
# PrototypeMemory
# ═══════════════════════════════════════════════════════════════════

def test_memory_add_get_has_classes():
    """基本存取接口 | basic add/get/has/classes."""
    mem = PrototypeMemory(embed_dim=4)
    p = torch.tensor([1.0, 0.0, 0.0, 0.0])
    mem.add(5, p)
    assert mem.has(5) and not mem.has(6)
    assert mem.classes() == [5]
    assert torch.allclose(mem.get(5), p)


def test_memory_running_mean_normalized():
    """多次 add 同类 → running mean 且归一化 | repeated adds → normalized running mean."""
    mem = PrototypeMemory(embed_dim=2)
    mem.add(1, torch.tensor([1.0, 0.0]))
    mem.add(1, torch.tensor([0.0, 1.0]))
    got = mem.get(1)
    assert got.norm().item() == pytest.approx(1.0, abs=1e-6)
    # 两正交向量均值方向 = (1,1)/√2 | mean of two orthogonal unit vectors
    assert torch.allclose(got, torch.tensor([0.5, 0.5]).div(torch.tensor(0.5**2 * 2) ** 0.5), atol=1e-5)


def test_memory_missing_raises():
    """查询未登记类 → KeyError | querying an absent class raises."""
    mem = PrototypeMemory()
    with pytest.raises(KeyError):
        mem.get(99)


def test_memory_bad_shape_raises():
    """错误维度 → 报错 | wrong prototype dim raises."""
    mem = PrototypeMemory(embed_dim=256)
    with pytest.raises(ValueError):
        mem.add(1, torch.randn(128))
