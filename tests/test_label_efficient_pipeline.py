"""Tests for the independent non-episodic label-efficient segmentation pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adasam.datasets.industrial import LabelRatioSubset
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


def test_label_ratio_subsets_are_nested() -> None:
    dataset = list(range(1000))
    one = LabelRatioSubset(dataset, 1, seed=7)
    five = LabelRatioSubset(dataset, 5, seed=7)
    assert len(one) == 10
    assert len(five) == 50
    assert set(one.indices).issubset(five.indices)
