"""Tests for the spatial defect-aware DAPG-v2 path."""

from __future__ import annotations

import torch

from adasam.losses import LabelEfficientSegmentationLoss
from adasam.models import LabelEfficientSAM
from adasam.models.prompt import DefectAwarePromptGeneratorV2
from tests.test_defect_prompt import FakeFrozenBackbone, feature_batch


def test_dapg_v2_prompt_shapes_and_budget() -> None:
    generator = DefectAwarePromptGeneratorV2()
    prompts = generator(feature_batch())
    assert prompts["dense_prompt"].shape == (4, 256, 64, 64)
    assert prompts["token_prompt"].shape == (4, 8, 256)
    assert generator.parameter_count < 300_000


def test_dapg_v2_forward_backward_and_frozen_backbone() -> None:
    model = LabelEfficientSAM(
        FakeFrozenBackbone(), num_classes=4, decoder_dim=32, prompt_version="v2", num_prompt=8
    )
    image = torch.rand(4, 3, 128, 128)
    logits, prompts = model.forward_with_prompts(image)
    assert logits.shape == (4, 4, 128, 128)
    assert prompts is not None
    assert prompts["dense_prompt"].shape == (4, 256, 64, 64)
    target = torch.randint(0, 4, (4, 128, 128))
    LabelEfficientSegmentationLoss()(logits, target).backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.prompt_generator.parameters())
    assert any(parameter.grad is not None for parameter in model.decoder.parameters())


def test_dapg_v2_increment_is_under_300k() -> None:
    baseline = LabelEfficientSAM(FakeFrozenBackbone(), num_classes=4, decoder_dim=96)
    dapg_v2 = LabelEfficientSAM(
        FakeFrozenBackbone(), num_classes=4, decoder_dim=96, prompt_version="v2"
    )
    increment = dapg_v2.parameter_counts()["trainable"] - baseline.parameter_counts()["trainable"]
    assert increment < 300_000
