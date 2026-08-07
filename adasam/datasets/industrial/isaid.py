"""Semantic iSAID loader for the prepared official train/val/test layout."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class ISAIDSemanticDataset(Dataset):
    """Load iSAID semantic masks (RGB palette) as 16-class masks.

    Expected root layout is the official extracted dataset:
    ``TrainData/train``, ``ValidationData/val`` and ``TestData/TestData``.
    Train/val images are expected at ``<split>/images/images`` and semantic
    masks at ``<split>/Semantic_masks/images/images``.
    """

    NUM_CLASSES = 16
    IGNORE_INDEX = 255
    CLASS_NAMES = [
        "background", "small_vehicle", "large_vehicle", "plane",
        "storage_tank", "ship", "harbor", "ground_track_field",
        "soccer_ball_field", "tennis_court", "swimming_pool",
        "baseball_diamond", "basketball_court", "bridge", "helicopter",
        "roundabout",
    ]
    # RGB palette used by the iSAID semantic mask release.
    PALETTE = np.array([
        [0, 0, 0], [0, 0, 63], [0, 63, 63], [0, 63, 0],
        [0, 63, 127], [0, 63, 191], [0, 63, 255], [0, 127, 63],
        [0, 127, 127], [0, 0, 127], [0, 0, 191], [0, 0, 255],
        [0, 127, 191], [0, 127, 255], [0, 191, 127], [0, 191, 191],
    ], dtype=np.uint8)

    def __init__(self, root: str | Path, split: str = "train", image_size: int | tuple[int, int] = 800, transforms=None):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.root = Path(root)
        self.split = split
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.transforms = transforms
        if split == "train":
            base = self.root / "TrainData" / "train"
        elif split == "val":
            base = self.root / "ValidationData" / "val"
        else:
            base = self.root / "TestData" / "TestData"
        if split == "test":
            image_paths = sorted((base / "Part1-001" / "images").glob("*.png"))
            image_paths += sorted((base / "Part1-002" / "images").glob("*.png"))
            self.mask_dir = None
        else:
            self.image_dir = base / "images" / "images"
            self.mask_dir = base / "Semantic_masks" / "images" / "images"
            image_paths = sorted(self.image_dir.glob("*.png"))
            if not self.image_dir.exists() or not image_paths:
                raise FileNotFoundError(f"iSAID {split} images not found: {self.image_dir}")
            if not self.mask_dir.exists():
                raise FileNotFoundError(f"iSAID {split} semantic masks not found: {self.mask_dir}")
        if not image_paths:
            raise RuntimeError(f"No iSAID images found for split={split} under {base}")
        self.samples = [(p, self._mask_path(p) if split != "test" else None, p.stem) for p in image_paths]

    def _mask_path(self, image_path: Path) -> Path:
        return self.mask_dir / f"{image_path.stem}_instance_color_RGB.png"

    def __len__(self) -> int:
        return len(self.samples)

    @classmethod
    def _decode_mask(cls, mask: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) if mask.ndim == 3 else mask[..., None]
        result = np.full(rgb.shape[:2], cls.IGNORE_INDEX, dtype=np.int64)
        for class_id, color in enumerate(cls.PALETTE):
            result[np.all(rgb == color, axis=-1)] = class_id
        return result

    def __getitem__(self, index: int) -> dict:
        image_path, mask_path, sample_id = self.samples[index]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        image = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1)
        image = F.interpolate(image[None], self.image_size, mode="bilinear", align_corners=False)[0]
        sample = {"image": image, "image_id": sample_id, "image_size": self.image_size}
        if mask_path is not None:
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing semantic mask for {image_path}: {mask_path}")
            mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
            mask = torch.from_numpy(self._decode_mask(mask_bgr))[None]
            mask = F.interpolate(mask.float()[None], self.image_size, mode="nearest")[0].long()
            sample["masks"] = mask
            if self.transforms is not None:
                sample = self.transforms(sample)
            return {"image": sample["image"], "id": sample_id, "mask": sample["masks"].squeeze(0).long()}
        return {"image": image, "id": sample_id}
