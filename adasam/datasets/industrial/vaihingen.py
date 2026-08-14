"""ISPRS Vaihingen semantic segmentation dataset for pre-cut patches."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class VaihingenSemanticDataset(Dataset):
    """Load paired ``images_1024`` TIFFs and indexed ``masks_1024`` PNGs.

    The source used by this project stores an additional special/boundary label
    as value 6. Values 0..5 are the six semantic classes; value 6 is mapped to
    ``IGNORE_INDEX`` rather than counted as clutter.
    """

    NUM_CLASSES = 6
    IGNORE_INDEX = 255
    CLASS_NAMES = [
        "impervious_surface",
        "building",
        "low_vegetation",
        "tree",
        "car",
        "clutter_background",
    ]

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int | tuple[int, int] = 512,
        transforms=None,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be one of: train, test")
        self.root = Path(root)
        self.split = split
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.transforms = transforms
        split_root = self.root / split
        image_dir = split_root / "images_1024"
        mask_dir = split_root / "masks_1024"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(
                f"Expected Vaihingen directories {image_dir} and {mask_dir}"
            )

        self.samples: list[tuple[Path, Path, int]] = []
        for image_path in sorted(image_dir.glob("*.tif")):
            mask_path = mask_dir / f"{image_path.stem}.png"
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
            match = re.search(r"area(\d+)", image_path.stem)
            if match is None:
                raise ValueError(f"Cannot determine source area from {image_path.name}")
            self.samples.append((image_path, mask_path, int(match.group(1))))
        if not self.samples:
            raise RuntimeError(f"No Vaihingen TIFF patches found in {image_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        image_path, mask_path, area_id = self.samples[index]
        image_array = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask_array = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image_array is None:
            raise ValueError(f"Cannot read Vaihingen image: {image_path}")
        if mask_array is None or mask_array.ndim != 2:
            raise ValueError(f"Cannot read indexed Vaihingen mask: {mask_path}")

        # OpenCV reads the TIFF channel container as BGR. The underlying source is
        # Vaihingen IRRG unless the dataset provider explicitly converted it to RGB.
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = torch.from_numpy(image_array).permute(2, 0, 1).float()
        image = F.interpolate(
            image.unsqueeze(0), self.image_size, mode="bilinear", align_corners=False
        ).squeeze(0)

        raw_mask = torch.from_numpy(mask_array.astype(np.int64))
        mask = torch.full_like(raw_mask, self.IGNORE_INDEX)
        valid = (raw_mask >= 0) & (raw_mask <= 5)
        mask[valid] = raw_mask[valid]
        mask = F.interpolate(
            mask[None, None].float(), self.image_size, mode="nearest"
        ).squeeze(0).long()

        sample = {
            "image": image,
            "masks": mask,
            "image_id": image_path.stem,
            "image_size": self.image_size,
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        return {
            "image": sample["image"],
            "mask": sample["masks"].squeeze(0).long(),
            "id": image_path.stem,
            "area_id": area_id,
        }
