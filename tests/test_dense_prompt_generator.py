"""
DensePromptGenerator 测试 | DPG tests.
========================================

纯合成张量, 不依赖权重与数据 | Synthetic tensors only; no weights/data required.

v2: support_memory [M, C] 替代 prototype [C]; 新增 dense_prompt 输出.
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


def _make_support_memory(m: int = 16, c: int = 32) -> torch.Tensor:
    """Create synthetic support memory tokens [M, C]."""
    torch.manual_seed(2)
    return torch.randn(m, c)


def _make_inputs(c: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    feats = torch.randn(1, c, GH, GW)
    support_mem = _make_support_memory(16, c)  # [M, C]
    pe = torch.randn(1, c, GH, GW)
    return feats, support_mem, pe


def test_output_shapes():
    dpg = _make_dpg()
    feats, support_mem, pe = _make_inputs()
    out = dpg(feats, support_mem, pe)
    assert isinstance(out, DPGOutput)
    assert out.instance_queries.shape == (8, 32)
    assert out.objectness_logits.shape == (8,)
    assert out.mask_logits.shape == (8, GH, GW)
    assert len(out.aux) == 2
    for a in out.aux:
        assert a["mask_logits"].shape == (8, GH, GW)
        assert a["objectness_logits"].shape == (8,)
    # v2: dense_prompt should be generated when support_memory is non-empty
    assert out.dense_prompt is not None
    assert out.dense_prompt.shape == (1, 32, 1, 1)


def test_dense_prompt_zero_init():
    """Zero-init: dense_prompt ≈ 0 at initialization (identity start)."""
    dpg = _make_dpg()
    feats, support_mem, pe = _make_inputs()
    dpg.eval()
    with torch.no_grad():
        out = dpg(feats, support_mem, pe)
    # dense_prompt_gen is zero-initialized → near-zero output
    assert out.dense_prompt.abs().max().item() < 1e-4


def test_empty_support_memory():
    """Empty support memory [0, C] → dense_prompt=None, no crash."""
    dpg = _make_dpg()
    feats, _, pe = _make_inputs()
    empty_mem = torch.zeros(0, 32)
    out = dpg(feats, empty_mem, pe)
    assert out.dense_prompt is None
    assert out.instance_queries.shape == (8, 32)
    assert out.objectness_logits.shape == (8,)


def test_degenerate_attn_mask_guard_no_nan():
    """全阻断行守卫: 极端负掩码预测下无 NaN | all-blocked guard: no NaN under extreme masks."""
    dpg = _make_dpg()
    feats, support_mem, pe = _make_inputs()
    # Force extremely negative masks → all rows blocked
    with torch.no_grad():
        for layer in dpg.mask_embed.layers:
            layer.weight.zero_()
            layer.bias.fill_(-100.0)
    out = dpg(feats, support_mem, pe)
    assert torch.isfinite(out.instance_queries).all()
    assert torch.isfinite(out.objectness_logits).all()
    assert torch.isfinite(out.mask_logits).all()


def test_gradients_flow():
    """Gradients reach query_feat / mask_embed / dense_prompt_gen."""
    dpg = _make_dpg()
    feats, support_mem, pe = _make_inputs()
    out = dpg(feats, support_mem, pe)
    loss = out.mask_logits.sum() + out.objectness_logits.sum()
    if out.dense_prompt is not None:
        loss = loss + out.dense_prompt.sum()
    for a in out.aux:
        loss = loss + a["mask_logits"].sum()
    loss.backward()
    assert dpg.query_feat.weight.grad is not None
    assert dpg.query_feat.weight.grad.abs().sum() > 0
    # v2: dense_prompt_gen (not proto_proj) receives gradients
    assert dpg.dense_prompt_gen[-1].weight.grad is not None
    assert dpg.dense_prompt_gen[-1].bias.grad is not None
    assert dpg.mask_embed.layers[0].weight.grad is not None


def test_deterministic_forward():
    """dropout=0 时前向确定 | forward is deterministic with dropout=0."""
    dpg = _make_dpg()
    feats, support_mem, pe = _make_inputs()
    dpg.eval()
    with torch.no_grad():
        a = dpg(feats, support_mem, pe)
        b = dpg(feats, support_mem, pe)
    assert torch.equal(a.instance_queries, b.instance_queries)
    assert torch.equal(a.objectness_logits, b.objectness_logits)


def test_config_from_dict_ignores_unknown_keys():
    cfg = DensePromptGeneratorConfig.from_dict(
        {"num_queries": 16, "num_layers": 1, "legacy_key": 123}
    )
    assert cfg.num_queries == 16
    assert cfg.num_layers == 1
    assert cfg.embed_dim == 256  # 默认 | default
