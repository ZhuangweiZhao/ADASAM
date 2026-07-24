"""
SemanticPriorGenerator 测试 | SPG tests.
===========================================

纯合成张量, 不依赖权重与数据 | Synthetic tensors only; no weights/data required.

v5 (protocol-aligned): SPG 只暴露 semantic_prior + prior_mask + prior_aux,
不再生产 dense_prompt / sparse_token。
"""

from __future__ import annotations

import torch

from adasam.prompt import SemanticPriorGenerator, SemanticPriorGeneratorConfig, SPGOutput

GH = GW = 16  # 小网格加速测试 (真实为 64) | small grid for speed (64 in production)


def _make_spg(**kwargs) -> SemanticPriorGenerator:
    cfg = SemanticPriorGeneratorConfig(
        num_probes=kwargs.pop("num_probes", 8),
        embed_dim=kwargs.pop("embed_dim", 32),
        num_layers=kwargs.pop("num_layers", 2),
        num_heads=kwargs.pop("num_heads", 4),
        ffn_dim=kwargs.pop("ffn_dim", 64),
        **kwargs,
    )
    torch.manual_seed(0)
    return SemanticPriorGenerator(cfg)


def _make_support_memory(m: int = 16, c: int = 32) -> torch.Tensor:
    torch.manual_seed(2)
    return torch.randn(m, c)


def _make_inputs(c: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    feats = torch.randn(1, c, GH, GW)
    support_mem = _make_support_memory(16, c)
    pe = torch.randn(1, c, GH, GW)
    return feats, support_mem, pe


def test_output_shapes():
    """SPG outputs unified prior (no dense_prompt/sparse_token)."""
    spg = _make_spg()
    feats, support_mem, pe = _make_inputs()

    out = spg(feats, support_mem, pe)
    assert isinstance(out, SPGOutput)
    # Core outputs only
    assert out.semantic_prior.shape == (1, 32, GH, GW)
    assert out.prior_mask.shape == (1, 1, GH, GW)
    # prior_aux stores unified prior_mask [1, gh, gw] (NOT per-probe [N, gh, gw])
    assert len(out.prior_aux) == 2  # num_layers=2
    for a in out.prior_aux:
        assert a["prior_mask"].shape == (1, GH, GW)  # unified, not per-probe
        assert "probe_logits" not in a  # no longer exposed


def test_empty_support_memory():
    """Empty support memory [0, C] → no crash, semantic_prior still generated."""
    spg = _make_spg()
    feats, _, pe = _make_inputs()
    empty_mem = torch.zeros(0, 32)
    out = spg(feats, empty_mem, pe)
    assert out.semantic_prior.shape == (1, 32, GH, GW)
    assert out.prior_mask.shape == (1, 1, GH, GW)


def test_degenerate_attn_mask_guard_no_nan():
    """全阻断行守卫: 极端负掩码预测下无 NaN."""
    spg = _make_spg()
    feats, support_mem, pe = _make_inputs()
    with torch.no_grad():
        for layer in spg.probe_proj.layers:
            layer.weight.zero_()
            layer.bias.fill_(-100.0)
    out = spg(feats, support_mem, pe)
    assert torch.isfinite(out.semantic_prior).all()
    assert torch.isfinite(out.prior_mask).all()


def test_gradients_flow():
    """Gradients reach probe_feat / probe_proj / prior_head."""
    spg = _make_spg()
    feats, support_mem, pe = _make_inputs()
    out = spg(feats, support_mem, pe)

    loss = (out.semantic_prior.sum() + out.prior_mask.sum())
    for a in out.prior_aux:
        loss = loss + a["prior_mask"].sum()
    loss.backward()

    # Probe parameters
    assert spg.probe_feat.weight.grad is not None
    assert spg.probe_feat.weight.grad.abs().sum() > 0
    # Probe projection
    assert spg.probe_proj.layers[0].weight.grad is not None
    # Prior heads
    # prior_head (last conv layer)
    assert spg.prior_head[-1].weight.grad is not None
    # prior_mask_head
    assert spg.prior_mask_head[-1].weight.grad is not None


def test_deterministic_forward():
    """dropout=0 时前向确定."""
    spg = _make_spg()
    feats, support_mem, pe = _make_inputs()
    spg.eval()
    with torch.no_grad():
        a = spg(feats, support_mem, pe)
        b = spg(feats, support_mem, pe)
    assert torch.equal(a.semantic_prior, b.semantic_prior)
    assert torch.equal(a.prior_mask, b.prior_mask)


def test_config_from_dict_accepts_legacy_num_queries():
    """Legacy key 'num_queries' is accepted as alias for 'num_probes'."""
    cfg = SemanticPriorGeneratorConfig.from_dict(
        {"num_queries": 16, "num_layers": 1, "legacy_key": 123}
    )
    assert cfg.num_probes == 16  # mapped from num_queries
    assert cfg.num_layers == 1
    assert cfg.embed_dim == 256


def test_config_from_dict_with_num_probes():
    """New key 'num_probes' works correctly."""
    cfg = SemanticPriorGeneratorConfig.from_dict(
        {"num_probes": 32, "num_layers": 4}
    )
    assert cfg.num_probes == 32
    assert cfg.num_layers == 4


def test_prior_aux_is_unified():
    """prior_aux stores unified [1, gh, gw] masks, not per-probe [N, gh, gw]."""
    spg = _make_spg(num_probes=6, num_layers=3)
    feats, support_mem, pe = _make_inputs()
    out = spg(feats, support_mem, pe)

    assert len(out.prior_aux) == 3
    for a in out.prior_aux:
        mask = a["prior_mask"]
        assert mask.ndim == 3  # [1, gh, gw]
        assert mask.shape[0] == 1  # unified, not per-probe N
        assert mask.shape[1] == GH
        assert mask.shape[2] == GW
