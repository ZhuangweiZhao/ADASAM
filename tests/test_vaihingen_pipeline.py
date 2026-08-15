from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from adasam.datasets.industrial import VaihingenSemanticDataset
from tools.train_vaihingen import add_official_metrics, build_model, class_weights


def _make_sample(root: Path, split: str = "train") -> None:
    image_dir = root / split / "images_1024"
    mask_dir = root / split / "masks_1024"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[..., 1] = 128
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[:4, :4] = 4
    mask[4:8, 4:8] = 5
    mask[8:12, 8:12] = 6
    cv2.imwrite(str(image_dir / "top_mosaic_09cm_area11_0_0.tif"), image)
    cv2.imwrite(str(mask_dir / "top_mosaic_09cm_area11_0_0.png"), mask)


def test_vaihingen_loads_resizes_and_ignores_special_label(tmp_path: Path) -> None:
    _make_sample(tmp_path)
    dataset = VaihingenSemanticDataset(tmp_path, image_size=32)
    sample = dataset[0]
    assert sample["image"].shape == (3, 32, 32)
    assert sample["mask"].shape == (32, 32)
    assert set(torch.unique(sample["mask"]).tolist()) == {0, 4, 5, 255}
    assert sample["area_id"] == 11
    assert sample["id"] == "top_mosaic_09cm_area11_0_0"


def test_vaihingen_rejects_missing_pair(tmp_path: Path) -> None:
    image_dir = tmp_path / "train" / "images_1024"
    (tmp_path / "train" / "masks_1024").mkdir(parents=True)
    image_dir.mkdir(parents=True)
    cv2.imwrite(str(image_dir / "top_mosaic_09cm_area1_0_0.tif"), np.zeros((4, 4, 3), np.uint8))
    try:
        VaihingenSemanticDataset(tmp_path)
    except FileNotFoundError as error:
        assert "Missing mask" in str(error)
    else:
        raise AssertionError("missing mask must be rejected")


def test_vaihingen_class_weights_ignore_special_label(tmp_path: Path) -> None:
    _make_sample(tmp_path)
    dataset = VaihingenSemanticDataset(tmp_path, image_size=16)
    weights = class_weights(dataset, [0])
    assert weights.shape == (6,)
    assert torch.isfinite(weights).all()


def test_official_metrics_exclude_clutter() -> None:
    result = add_official_metrics({
        "per_class_IoU": [0.5, 0.6, 0.7, 0.8, 0.9, 0.0],
        "per_class_Dice": [0.6, 0.7, 0.8, 0.9, 1.0, 0.0],
    })
    assert result["mIoU_5"] == pytest.approx(0.7)
    assert result["mean_F1_5"] == pytest.approx(0.8)
    assert len(result["official_classes"]) == 5


def test_vaihingen_builds_segformer_baseline() -> None:
    args = type("Args", (), {
        "model": "segformer",
        "lora_rank": 0,
        "pretrained": False,
        "baseline_encoder": "resnet50",
        "segformer_variant": "b0",
    })()
    model = build_model(args, torch.device("cpu"))
    with torch.no_grad():
        output = model(torch.rand(1, 3, 64, 96))
    assert output.shape == (1, VaihingenSemanticDataset.NUM_CLASSES, 64, 96)


def test_vaihingen_rejects_lora_for_standard_baseline() -> None:
    args = type("Args", (), {
        "model": "segformer",
        "lora_rank": 4,
        "pretrained": False,
        "baseline_encoder": "resnet50",
        "segformer_variant": "b0",
    })()
    with pytest.raises(ValueError, match="LoRA"):
        build_model(args, torch.device("cpu"))
