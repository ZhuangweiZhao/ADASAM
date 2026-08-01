"""Tests for the Defect-aware Prompt Generator (DAPG)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adasam.losses import LabelEfficientSegmentationLoss
from adasam.models import LabelEfficientSAM
from adasam.models.prompt import DefectPromptGenerator


class FakeFrozenBackbone(nn.Module):
    img_size = 224

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 256, 1)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = F.adaptive_avg_pool2d(self.stem(image), (14, 14))
        return {
            "P3": F.adaptive_avg_pool2d(embedding[:, :128], (28, 28)),
            "P4": F.adaptive_avg_pool2d(embedding[:, :160], (14, 14)),
            "embedding": embedding,
        }


def feature_batch(batch: int = 4) -> dict[str, torch.Tensor]:
    return {
        "P3": torch.randn(batch, 128, 28, 28),
        "P4": torch.randn(batch, 160, 14, 14),
        "embedding": torch.randn(batch, 256, 14, 14),
    }


def test_prompt_shape_and_parameter_budget() -> None:
    generator = DefectPromptGenerator()
    prompts = generator(feature_batch())
    assert prompts.shape == (4, 16, 256)
    assert generator.parameter_count < 100_000


def test_dapg_model_forward_backward_and_frozen_backbone() -> None:
    model = LabelEfficientSAM(
        FakeFrozenBackbone(), num_classes=4, decoder_dim=32, use_dapg=True
    )
    image = torch.rand(4, 3, 128, 128)
    logits, prompts = model.forward_with_prompts(image)
    assert logits.shape == (4, 4, 128, 128)
    assert prompts is not None and prompts.shape == (4, 16, 256)
    target = torch.randint(0, 4, (4, 128, 128))
    LabelEfficientSegmentationLoss()(logits, target).backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.prompt_generator.parameters())
    assert any(parameter.grad is not None for parameter in model.decoder.parameters())


def test_dapg_increment_is_below_200k_and_baseline_is_unchanged() -> None:
    baseline = LabelEfficientSAM(FakeFrozenBackbone(), num_classes=4, decoder_dim=96)
    dapg = LabelEfficientSAM(
        FakeFrozenBackbone(), num_classes=4, decoder_dim=96, use_dapg=True
    )
    increment = dapg.parameter_counts()["trainable"] - baseline.parameter_counts()["trainable"]
    assert increment < 200_000
    assert baseline.prompt_generator is None
