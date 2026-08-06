from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from adasam.datasets.industrial import VisASemanticDataset


def make_visa_fixture(root: Path) -> None:
    (root / "split_csv").mkdir(parents=True)
    (root / "toy" / "Data" / "Images" / "Normal").mkdir(parents=True)
    (root / "toy" / "Data" / "Images" / "Anomaly").mkdir(parents=True)
    (root / "toy" / "Data" / "Masks" / "Anomaly").mkdir(parents=True)
    cv2.imwrite(str(root / "toy" / "Data" / "Images" / "Normal" / "n.JPG"), np.zeros((12, 16, 3), np.uint8))
    cv2.imwrite(str(root / "toy" / "Data" / "Images" / "Anomaly" / "a.JPG"), np.full((12, 16, 3), 127, np.uint8))
    mask = np.zeros((12, 16), np.uint8)
    mask[2:6, 3:8] = 255
    cv2.imwrite(str(root / "toy" / "Data" / "Masks" / "Anomaly" / "a.JPG"), mask)
    with (root / "split_csv" / "1cls.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["object", "split", "label", "image", "mask"])
        writer.writerow(["toy", "train", "normal", "toy/Data/Images/Normal/n.JPG", ""])
        writer.writerow(["toy", "test", "anomaly", "toy/Data/Images/Anomaly/a.JPG", "toy/Data/Masks/Anomaly/a.JPG"])


def test_visa_loader_maps_binary_masks_and_resizes(tmp_path: Path) -> None:
    make_visa_fixture(tmp_path)
    train = VisASemanticDataset(tmp_path, "train", image_size=32)
    test = VisASemanticDataset(tmp_path, "test", image_size=32)
    assert len(train) == 1 and len(test) == 1
    assert train[0]["image"].shape == (3, 32, 32)
    assert torch.equal(torch.unique(train[0]["mask"]), torch.tensor([0]))
    assert set(torch.unique(test[0]["mask"]).tolist()) == {0, 1}
    assert test[0]["anomaly"] == 1


def test_visa_all_split_contains_normal_and_anomaly(tmp_path: Path) -> None:
    make_visa_fixture(tmp_path)
    dataset = VisASemanticDataset(tmp_path, "all")
    assert len(dataset) == 2
    assert {sample["label"] for sample in dataset.samples} == {"normal", "anomaly"}
