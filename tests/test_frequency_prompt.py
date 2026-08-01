"""Tests for the frequency-aware FDAPG-v3 path."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adasam.losses import LabelEfficientSegmentationLoss
from adasam.models import LabelEfficientSAM
from adasam.models.prompt import FrequencyAwareDefectPromptGenerator


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


def test_frequency_prompt_shapes_finite_and_budget() -> None:
    generator = FrequencyAwareDefectPromptGenerator()
    prompts = generator(feature_batch())
    assert prompts["dense_prompt"].shape == (4, 256, 64, 64)
    assert prompts["token_prompt"].shape == (4, 8, 256)
    assert prompts["frequency_heatmap"].shape == (4, 1, 64, 64)
    assert prompts["dense_activation"].shape == (4, 1, 64, 64)
    assert all(torch.isfinite(value).all() for value in prompts.values())
    assert generator.parameter_count < 300_000


def test_frequency_prompt_backward() -> None:
    generator = FrequencyAwareDefectPromptGenerator()
    prompts = generator(feature_batch())
    loss = prompts["dense_prompt"].square().mean() + prompts["token_prompt"].square().mean()
    loss.backward()
    assert generator.alpha.grad is not None
    assert torch.isfinite(generator.alpha.grad)
    assert any(parameter.grad is not None for parameter in generator.parameters())


def test_fdapg_v3_model_forward_backward_and_frozen_backbone() -> None:
    model = LabelEfficientSAM(
        FakeFrozenBackbone(), num_classes=4, decoder_dim=32, prompt_version="v3", num_prompt=8
    )
    image = torch.rand(4, 3, 128, 128)
    logits, prompts = model.forward_with_prompts(image)
    assert logits.shape == (4, 4, 128, 128)
    assert prompts is not None
    target = torch.randint(0, 4, (4, 128, 128))
    LabelEfficientSegmentationLoss()(logits, target).backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.decoder.parameters())


def test_fdapg_v3_increment_is_under_300k() -> None:
    baseline = LabelEfficientSAM(FakeFrozenBackbone(), num_classes=4, decoder_dim=96)
    fdapg_v3 = LabelEfficientSAM(
        FakeFrozenBackbone(), num_classes=4, decoder_dim=96, prompt_version="v3", num_prompt=8
    )
    increment = fdapg_v3.parameter_counts()["trainable"] - baseline.parameter_counts()["trainable"]
    assert increment < 300_000
