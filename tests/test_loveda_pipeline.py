from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from adasam.datasets.industrial import LoveDASemanticDataset, fixed_validation_split_indices
from adasam.losses import LabelEfficientSegmentationLoss
from tools.train_loveda import collect_routing_statistics


def make_loveda_sample(root: Path, split: str = "Train") -> None:
    image_dir = root / split / split / "Rural" / "images_png"
    mask_dir = root / split / split / "Rural" / "masks_png"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[..., 1] = 128
    mask = np.ones((16, 16), dtype=np.uint8)
    mask[:4, :4] = 0
    mask[4:8, 4:8] = 7
    cv2.imwrite(str(image_dir / "0.png"), image)
    cv2.imwrite(str(mask_dir / "0.png"), mask)


def test_loveda_dataset_maps_labels_and_resizes(tmp_path: Path) -> None:
    make_loveda_sample(tmp_path)
    dataset = LoveDASemanticDataset(tmp_path, "train", image_size=32)
    sample = dataset[0]
    assert sample["image"].shape == (3, 32, 32)
    assert sample["mask"].shape == (32, 32)
    assert set(torch.unique(sample["mask"]).tolist()) == {0, 6, 255}
    assert sample["id"] == "Rural_0"


def test_loveda_loss_ignores_unlabeled_pixels() -> None:
    prediction = torch.randn(2, 7, 16, 16, requires_grad=True)
    target = torch.randint(0, 7, (2, 16, 16))
    target[:, :4, :4] = 255
    loss = LabelEfficientSegmentationLoss(ignore_index=255)(prediction, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad is not None


def test_loveda_fixed_split_is_nested() -> None:
    train_5, validation_5, pool = fixed_validation_split_indices(2522, 5, 42, 0.2, 42)
    train_10, validation_10, _ = fixed_validation_split_indices(2522, 10, 42, 0.2, 42)
    assert len(validation_5) == 504
    assert len(pool) == 2018
    assert len(train_5) == 101
    assert len(train_10) == 202
    assert validation_5 == validation_10
    assert set(train_5).issubset(train_10)


def test_routing_statistics_accepts_no_ignore_index() -> None:
    class Decoder:
        fusion_version = "scsr"
        representation_budget = 3
        last_routing = {
            "weights": torch.full((1, 3, 2, 2), 1.0 / 3.0),
            "entropy": torch.zeros(1, 2, 2),
        }

    class Model:
        decoder = Decoder()

        def eval(self):
            return self

        def __call__(self, image):
            return image

    loader = [
        {
            "image": torch.zeros(1, 3, 4, 4),
            "mask": torch.tensor(
                [[[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]]]
            ),
        }
    ]

    statistics = collect_routing_statistics(
        Model(), loader, torch.device("cpu"), num_classes=2, ignore_index=None
    )

    assert statistics is not None
    assert statistics["pixels"] == 4
    assert set(statistics["class_mean_weights"]) == {"0", "1"}
