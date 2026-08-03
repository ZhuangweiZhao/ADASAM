"""LoveDA semantic segmentation dataset for the label-efficient protocol."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class LoveDASemanticDataset(Dataset):
    """Load LoveDA Rural/Urban images and map labels 1..7 to 0..6."""

    NUM_CLASSES = 7
    IGNORE_INDEX = 255
    CLASS_NAMES = [
        "background",
        "building",
        "road",
        "water",
        "barren",
        "forest",
        "agriculture",
    ]

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int | tuple[int, int] = 512,
        transforms=None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")
        self.root = Path(root)
        self.split = split
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.transforms = transforms
        split_name = {"train": "Train", "val": "Val", "test": "Test"}[split]
        split_root = self.root / split_name / split_name
        if not split_root.exists():
            raise FileNotFoundError(f"LoveDA split directory not found: {split_root}")
        self.samples: list[tuple[Path, Path | None, str]] = []
        for domain in ("Rural", "Urban"):
            image_dir = split_root / domain / "images_png"
            mask_dir = split_root / domain / "masks_png"
            for image_path in sorted(image_dir.glob("*.png"), key=lambda path: int(path.stem)):
                mask_path = mask_dir / image_path.name
                if split != "test" and not mask_path.exists():
                    continue
                self.samples.append((image_path, mask_path if mask_path.exists() else None, f"{domain}_{image_path.stem}"))
        if not self.samples:
            raise RuntimeError(f"No LoveDA samples found in {split_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        image_path, mask_path, sample_id = self.samples[index]
        image_array = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_array is None:
            raise ValueError(f"Cannot read LoveDA image: {image_path}")
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = torch.from_numpy(image_array).permute(2, 0, 1).float()
        image = F.interpolate(
            image.unsqueeze(0), self.image_size, mode="bilinear", align_corners=False
        ).squeeze(0)
        sample = {"image": image, "image_id": sample_id, "image_size": self.image_size}
        if mask_path is not None:
            mask_array = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask_array is None:
                raise ValueError(f"Cannot read LoveDA mask: {mask_path}")
            raw_mask = torch.from_numpy(mask_array.astype(np.int64))
            mask = torch.full_like(raw_mask, self.IGNORE_INDEX)
            valid = (raw_mask >= 1) & (raw_mask <= self.NUM_CLASSES)
            mask[valid] = raw_mask[valid] - 1
            mask = F.interpolate(
                mask[None, None].float(), self.image_size, mode="nearest"
            ).squeeze(0).long()
            sample["masks"] = mask
        if self.transforms is not None:
            if mask_path is None:
                raise ValueError("segmentation augmentation requires a LoveDA mask")
            sample = self.transforms(sample)
        result = {"image": sample["image"], "id": sample_id}
        if "masks" in sample:
            result["mask"] = sample["masks"].squeeze(0).long()
        return result
