"""
AdaSAMModel 端到端集成测试 | End-to-end integration tests for AdaSAMModel.
============================================================================

需 weights/mobile_sam.pt (缺失则 skip) | requires weights/mobile_sam.pt (skip if absent).

v5 (protocol-aligned): SPG outputs semantic_prior + prior_mask only.
Dense prompt / sparse token come from PromptFusion (or fallback).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from adasam.losses import SemanticSegLoss
from adasam.model import AdaSAMModel, AdaSAMModelConfig

_CKPT = Path(__file__).resolve().parents[1] / "weights" / "mobile_sam.pt"
_skip_ckpt = pytest.mark.skipif(not _CKPT.exists(), reason="weights/mobile_sam.pt not present")

_CFG = {
    "semantic_prior": {"num_probes": 8, "num_layers": 2, "ffn_dim": 256},
    "decoder": {},
    "support_encoder": {"n_support_tokens": 8, "n_memory_tokens": 32, "n_encoder_layers": 2},
    "geometric_prior": {"enabled": True},
    "prompt_fusion": {"enabled": True, "mode": "concat"},
}


def _make_support_data(k: int = 3):
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
    spg_out, low_res, iou_pred = model.forward_train(emb, sf, sm)

    # SPG unified outputs (no dense_prompt/sparse_token)
    assert spg_out.semantic_prior.shape == (1, 256, 64, 64)
    assert spg_out.prior_mask.shape == (1, 1, 64, 64)
    assert len(spg_out.prior_aux) == 2  # num_layers=2
    # prior_aux stores unified [1, gh, gw] masks
    for a in spg_out.prior_aux:
        assert a["prior_mask"].shape == (1, 64, 64)

    # SAM decoder: single mask output [1, 1, 256, 256]
    assert low_res.shape == (1, 1, 256, 256)
    assert iou_pred.shape == (1, 1)


@_skip_ckpt
def test_forward_train_to_criterion_backward(model):
    """Full chain: forward_train → semantic criterion → backward, gradients reach SPG."""
    emb = torch.randn(1, 256, 64, 64)
    sf, sm = _make_support_data(3)
    gt = torch.zeros(256, 256)
    gt[50:100, 50:100] = 1.0

    spg_out, low_res, iou_pred = model.forward_train(emb, sf, sm)

    # Build 2-channel FG/BG from single mask output
    fg_logits = low_res[0, 0]   # [256, 256]
    bg_logits = -fg_logits
    pred_2ch = torch.stack([bg_logits, fg_logits], dim=0).unsqueeze(0)

    # Gather unified prior masks for deep supervision
    prior_masks = []
    for aux_entry in spg_out.prior_aux:
        prior_masks.append(aux_entry["prior_mask"])  # [1, gh, gw]

    criterion = SemanticSegLoss()
    out = criterion(pred_2ch, gt.unsqueeze(0), prior_masks=prior_masks,
                    prior_mask=spg_out.prior_mask)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()

    # SPG probe parameters receive gradients
    assert model.spg.probe_feat.weight.grad is not None
    assert model.spg.probe_feat.weight.grad.abs().sum() > 0


@_skip_ckpt
def test_num_probes_property(model):
    """num_probes property reflects config."""
    assert model.num_probes == 8


@_skip_ckpt
def test_predict_output_shape(model):
    """predict() returns single mask [1, H, W] and score [1]."""
    emb = torch.randn(1, 256, 64, 64)
    sf, sm = _make_support_data(3)
    masks, scores = model.predict(emb, sf, sm, (1024, 1024), (256, 256), score_thr=0.0)
    assert masks.ndim == 3  # [1, H, W]
    assert masks.shape[0] == 1
    assert masks.shape[1] == 256
    assert scores.shape == (1,)
