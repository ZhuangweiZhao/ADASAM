"""
AdaSAMModel 端到端集成测试 | End-to-end integration tests for AdaSAMModel.
============================================================================

需 weights/mobile_sam.pt (缺失则 skip) | requires weights/mobile_sam.pt (skip if absent).

v2: support_features + support_masks replace prototype.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from adasam.losses import CriterionConfig, HungarianMatcher, MatcherConfig, SetCriterion
from adasam.model import AdaSAMModel, AdaSAMModelConfig

_CKPT = Path(__file__).resolve().parents[1] / "weights" / "mobile_sam.pt"
_skip_ckpt = pytest.mark.skipif(not _CKPT.exists(), reason="weights/mobile_sam.pt not present")

_CFG = {
    "prompt_generator": {"num_queries": 8, "num_layers": 2, "ffn_dim": 256},
    "decoder": {},
    "support_encoder": {"n_support_tokens": 8, "n_memory_tokens": 32, "n_encoder_layers": 2},
}


def _make_support_data(k: int = 3):
    """Create synthetic support features + masks."""
    sf = torch.randn(k, 256, 64, 64)
    sm = (torch.rand(k, 64, 64) > 0.3).float()
    return sf, sm


@pytest.fixture(scope="module")
def model():
    from adasam.backbone import build_mobile_sam
    sam = build_mobile_sam(_CKPT, device="cpu")
    torch.manual_seed(0)
    return AdaSAMModel(sam, AdaSAMModelConfig.from_dict(_CFG))


@_skip_ckpt
def test_forward_train_shapes(model):
    emb = torch.randn(1, 256, 64, 64)
    sf, sm = _make_support_data(3)
    dpg_out, low_res, iou_pred = model.forward_train(emb, sf, sm)
    assert dpg_out.instance_queries.shape == (8, 256)
    assert dpg_out.objectness_logits.shape == (8,)
    assert dpg_out.mask_logits.shape == (8, 64, 64)
    assert len(dpg_out.aux) == 2
    assert low_res.shape == (8, 1, 256, 256)
    assert iou_pred.shape == (8, 1)
    # v2: dense_prompt should be present when support is given
    assert dpg_out.dense_prompt is not None
    assert dpg_out.dense_prompt.shape == (1, 256, 1, 1)


@_skip_ckpt
def test_forward_train_to_criterion_backward(model):
    """Full chain: forward_train → criterion → backward, gradients reach DPG + SupportEncoder."""
    emb = torch.randn(1, 256, 64, 64)
    sf, sm = _make_support_data(3)
    gt = torch.zeros(3, 896, 896)
    for i in range(3):
        gt[i, i * 200 : i * 200 + 150, i * 200 : i * 200 + 150] = 1.0

    dpg_out, low_res, iou_pred = model.forward_train(emb, sf, sm)
    criterion = SetCriterion(HungarianMatcher(MatcherConfig()), CriterionConfig())
    out = criterion(low_res[:, 0], iou_pred[:, 0], dpg_out, gt)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()

    # DPG parameters receive gradients
    assert model.dpg.query_feat.weight.grad is not None
    assert model.dpg.query_feat.weight.grad.abs().sum() > 0

    # SupportEncoder parameters receive gradients
    # (mask_token only gets grads when a support needs padding; check
    #  memory_bank.memory_tokens instead which is always on the hot path)
    assert model.support_encoder.memory_bank.memory_tokens.grad is not None

    # SAM decoder parameters receive gradients
    any_sam_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.sam_decoder.mask_decoder.parameters()
    )
    assert any_sam_grad
    model.zero_grad(set_to_none=True)


@_skip_ckpt
def test_predict_contract(model):
    """predict → (masks bool [n,896,896], scores [n]∈[0,1]), n ≤ N | inference contract."""
    emb = torch.randn(1, 256, 64, 64)
    sf, sm = _make_support_data(3)
    masks, scores = model.predict(
        emb, sf, sm, input_size=(1024, 1024), original_size=(896, 896), score_thr=0.0
    )
    n = masks.shape[0]
    assert n <= model.num_queries
    assert masks.shape == (n, 896, 896) and masks.dtype == torch.bool
    assert scores.shape == (n,)
    if n:
        assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0


@_skip_ckpt
def test_predict_high_threshold_returns_empty(model):
    """score_thr=1.1 → empty output, no crash."""
    emb = torch.randn(1, 256, 64, 64)
    sf, sm = _make_support_data(3)
    masks, scores = model.predict(
        emb, sf, sm, input_size=(1024, 1024), original_size=(896, 896), score_thr=1.1
    )
    assert masks.shape == (0, 896, 896)
    assert scores.shape == (0,)


@_skip_ckpt
def test_state_dict_roundtrip(model):
    """Strict state_dict round-trip."""
    from adasam.backbone import build_mobile_sam
    sam2 = build_mobile_sam(_CKPT, device="cpu")
    model2 = AdaSAMModel(sam2, AdaSAMModelConfig.from_dict(_CFG))
    model2.load_state_dict(model.state_dict(), strict=True)
