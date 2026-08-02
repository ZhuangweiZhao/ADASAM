from __future__ import annotations

import torch

from adasam.datasets.augmentation import (
    BasicAugmentation,
    DefectAwareAugmentation,
    build_augmentation,
)


def sample() -> dict:
    image = torch.linspace(0.0, 1.0, 64 * 64).reshape(1, 64, 64).repeat(3, 1, 1)
    mask = torch.zeros(1, 64, 64, dtype=torch.long)
    mask[:, 8:16, 10:18] = 1
    mask[:, 30:38, 28:36] = 2
    return {"image": image, "masks": mask, "image_id": "synthetic", "image_size": (64, 64)}


def test_basic_augmentation_keeps_shape_and_classes() -> None:
    torch.manual_seed(7)
    original = sample()
    transform = BasicAugmentation(
        horizontal_flip_probability=1.0,
        vertical_flip_probability=1.0,
        rotation_probability=1.0,
    )
    augmented = transform(original)
    assert augmented["image"].shape == original["image"].shape
    assert augmented["masks"].shape == original["masks"].shape
    assert set(torch.unique(augmented["masks"]).tolist()) == {0, 1, 2}
    assert 0.0 <= float(augmented["image"].min()) <= float(augmented["image"].max()) <= 1.0


def test_defect_copy_paste_adds_bounded_pixels_without_losing_classes() -> None:
    torch.manual_seed(3)
    original = sample()
    before = int((original["masks"] > 0).sum())
    transform = DefectAwareAugmentation(
        copy_paste_probability=1.0,
        scale_range=(1.0, 1.0),
        max_added_area_ratio=0.15,
        horizontal_flip_probability=0.0,
        vertical_flip_probability=0.0,
        rotation_probability=0.0,
        brightness=0.0,
        contrast=0.0,
    )
    augmented = transform(original)
    after = int((augmented["masks"] > 0).sum())
    assert augmented["image"].shape == original["image"].shape
    assert augmented["masks"].shape == original["masks"].shape
    assert before < after <= before + round(64 * 64 * 0.15)
    assert set(torch.unique(augmented["masks"]).tolist()) == {0, 1, 2}
    assert torch.equal(original["masks"], sample()["masks"]), "augmentation mutated its input"


def test_none_mode_is_safe_for_validation_and_test() -> None:
    assert build_augmentation("none") is None
    assert isinstance(build_augmentation("basic"), BasicAugmentation)
    assert isinstance(build_augmentation("defect"), DefectAwareAugmentation)
