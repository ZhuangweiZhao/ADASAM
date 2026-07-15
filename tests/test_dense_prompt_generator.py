"""
DensePromptGenerator 测试 | DPG tests.
========================================

纯合成张量, 不依赖权重与数据 | Synthetic tensors only; no weights/data required.
"""

from __future__ import annotations

import torch

from adasam.prompt import DensePromptGenerator, DensePromptGeneratorConfig, DPGOutput

GH = GW = 16  # 小网格加速测试 (真实为 64) | small grid for speed (64 in production)


def _make_dpg(**kwargs) -> DensePromptGenerator:
    cfg = DensePromptGeneratorConfig(
        num_queries=kwargs.pop("num_queries", 8),
        embed_dim=kwargs.pop("embed_dim", 32),
        num_layers=kwargs.pop("num_layers", 2),
        num_heads=kwargs.pop("num_heads", 4),
        ffn_dim=kwargs.pop("ffn_dim", 64),
        **kwargs,
    )
    torch.manual_seed(0)
    return DensePromptGenerator(cfg)


def _make_inputs(c: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    feats = torch.randn(1, c, GH, GW)
    proto = torch.nn.functional.normalize(torch.randn(c), dim=0)
    pe = torch.randn(1, c, GH, GW)
    return feats, proto, pe


def test_output_shapes():
    dpg = _make_dpg()
    feats, proto, pe = _make_inputs()
    out = dpg(feats, proto, pe)
    assert isinstance(out, DPGOutput)
    assert out.instance_queries.shape == (8, 32)
    assert out.objectness_logits.shape == (8,)
    assert out.mask_logits.shape == (8, GH, GW)
    assert len(out.aux) == 2
    for a in out.aux:
        assert a["mask_logits"].shape == (8, GH, GW)
        assert a["objectness_logits"].shape == (8,)


def test_proto_conditioning_zero_init():
    """零初始化 → 初始时输出与原型无关 | zero-init → prototype-agnostic at init."""
    dpg = _make_dpg()
    feats, proto, pe = _make_inputs()
    dpg.eval()
    with torch.no_grad():
        out_a = dpg(feats, proto, pe)
        out_b = dpg(feats, torch.zeros_like(proto), pe)
    assert torch.allclose(out_a.instance_queries, out_b.instance_queries)
    assert torch.allclose(out_a.mask_logits, out_b.mask_logits)


def test_degenerate_attn_mask_guard_no_nan():
    """全阻断行守卫: 极端负掩码预测下无 NaN | all-blocked guard: no NaN under extreme masks."""
    dpg = _make_dpg()
    feats, proto, pe = _make_inputs()
    # 强制预测头输出极负掩码 → 所有行全阻断 | force extremely negative masks
    with torch.no_grad():
        for layer in dpg.mask_embed.layers:
            layer.weight.zero_()
            layer.bias.fill_(-100.0)
    out = dpg(feats, proto, pe)
    assert torch.isfinite(out.instance_queries).all()
    assert torch.isfinite(out.objectness_logits).all()
    assert torch.isfinite(out.mask_logits).all()


def test_gradients_flow():
    """梯度可达 query_feat / proto_proj / mask_embed | gradients reach key parameters."""
    dpg = _make_dpg()
    feats, proto, pe = _make_inputs()
    out = dpg(feats, proto, pe)
    loss = out.mask_logits.sum() + out.objectness_logits.sum()
    for a in out.aux:
        loss = loss + a["mask_logits"].sum()
    loss.backward()
    assert dpg.query_feat.weight.grad is not None
    assert dpg.query_feat.weight.grad.abs().sum() > 0
    assert dpg.proto_proj.weight.grad is not None
    assert dpg.proto_proj.weight.grad.abs().sum() > 0  # 原型非零 → 有梯度 | nonzero proto
    assert dpg.mask_embed.layers[0].weight.grad is not None


def test_deterministic_forward():
    """dropout=0 时前向确定 | forward is deterministic with dropout=0."""
    dpg = _make_dpg()
    feats, proto, pe = _make_inputs()
    dpg.eval()
    with torch.no_grad():
        a = dpg(feats, proto, pe)
        b = dpg(feats, proto, pe)
    assert torch.equal(a.instance_queries, b.instance_queries)
    assert torch.equal(a.objectness_logits, b.objectness_logits)


def test_config_from_dict_ignores_unknown_keys():
    cfg = DensePromptGeneratorConfig.from_dict(
        {"num_queries": 16, "num_layers": 1, "legacy_key": 123}
    )
    assert cfg.num_queries == 16
    assert cfg.num_layers == 1
    assert cfg.embed_dim == 256  # 默认 | default
