"""Tests for the independent non-episodic label-efficient segmentation pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adasam.datasets.industrial import LabelRatioSubset, fixed_validation_split_indices
from adasam.losses import LabelEfficientSegmentationLoss
from adasam.models import LabelEfficientSAM


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

    def train(self, mode: bool = True):
        return super().train(False)


def test_forward_512_and_batch_four_backward() -> None:
    model = LabelEfficientSAM(FakeFrozenBackbone(), num_classes=4, decoder_dim=32)
    image = torch.rand(4, 3, 512, 512)
    logits = model(image)
    assert logits.shape == (4, 4, 512, 512)
    target = torch.randint(0, 4, (4, 512, 512))
    LabelEfficientSegmentationLoss()(logits, target).backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.adapter.parameters())
    assert any(parameter.grad is not None for parameter in model.decoder.parameters())


def test_parameter_counts_partition_total() -> None:
    model = LabelEfficientSAM(FakeFrozenBackbone(), num_classes=4, decoder_dim=32)
    counts = model.parameter_counts()
    assert counts["total"] == counts["trainable"] + counts["frozen"]
    assert counts["trainable"] > 0
    assert counts["frozen"] > 0


def test_dapg_v2_dpm_batch_four_forward_backward() -> None:
    model = LabelEfficientSAM(
        FakeFrozenBackbone(),
        num_classes=4,
        decoder_dim=32,
        prompt_version="v2",
        num_prompt=8,
        prototype_version="dpm",
    )
    image = torch.rand(4, 3, 512, 512)
    target = torch.randint(0, 4, (4, 512, 512))
    logits, prompts, prototype_aux = model.forward_with_auxiliary(image, target)
    assert logits.shape == (4, 4, 512, 512)
    assert prompts is not None and prompts["dense_prompt"].ndim == 4
    assert prototype_aux is not None
    assert prototype_aux["similarity"].shape[:2] == (4, 4)
    LabelEfficientSegmentationLoss()(logits, target).backward()
    assert model.prototype_memory is not None
    assert model.prototype_memory.prior_projection.weight.grad is not None


def test_label_ratio_subsets_are_nested() -> None:
    dataset = list(range(1000))
    one = LabelRatioSubset(dataset, 1, seed=7)
    five = LabelRatioSubset(dataset, 5, seed=7)
    assert len(one) == 10
    assert len(five) == 50
    assert set(one.indices).issubset(five.indices)


def test_extended_label_ratios_are_nested() -> None:
    dataset = list(range(1000))
    ten = LabelRatioSubset(dataset, 10, seed=7)
    twenty = LabelRatioSubset(dataset, 20, seed=7)
    fifty = LabelRatioSubset(dataset, 50, seed=7)
    assert len(twenty) == 200
    assert len(fifty) == 500
    assert set(ten.indices).issubset(twenty.indices)
    assert set(twenty.indices).issubset(fifty.indices)


def test_fixed_validation_protocol_is_fixed_and_training_sets_are_nested() -> None:
    train_5, validation_5, pool_5 = fixed_validation_split_indices(1000, 5, seed=123)
    train_10, validation_10, pool_10 = fixed_validation_split_indices(1000, 10, seed=123)
    _, validation_other_seed, _ = fixed_validation_split_indices(1000, 10, seed=456)
    assert validation_5 == validation_10 == validation_other_seed
    assert len(validation_5) == 200
    assert len(pool_5) == len(pool_10) == 800
    assert len(train_5) == 40
    assert len(train_10) == 80
    assert set(train_5).issubset(train_10)
    assert set(train_10).isdisjoint(validation_10)
