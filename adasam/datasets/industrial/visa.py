"""VisA industrial anomaly segmentation dataset loader."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class VisASemanticDataset(Dataset):
    """Load VisA images with binary anomaly masks.

    The official VisA train split contains normal images only. Test anomaly masks
    are loaded when present; normal samples and train samples receive an all-zero
    mask. ``split='all'`` is useful for inspection, but must not be used for an
    unbiased test protocol.
    """

    NUM_CLASSES = 2
    IGNORE_INDEX = 255
    CLASS_NAMES = ["normal", "anomaly"]
    CSV_NAMES = {"1cls": "1cls.csv", "2cls_fewshot": "2cls_fewshot.csv", "2cls_highshot": "2cls_highshot.csv"}

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int | tuple[int, int] | None = None,
        transforms=None,
        split_csv: str = "1cls",
    ) -> None:
        if split not in {"train", "test", "all"}:
            raise ValueError("split must be one of: train, test, all")
        if split_csv not in self.CSV_NAMES:
            raise ValueError(f"split_csv must be one of: {sorted(self.CSV_NAMES)}")
        self.root = Path(root)
        csv_path = self.root / "split_csv" / self.CSV_NAMES[split_csv]
        if not csv_path.exists():
            raise FileNotFoundError(f"VisA split file not found: {csv_path}")
        self.split = split
        self.image_size = image_size
        self.transforms = transforms
        self.samples: list[dict[str, str | int]] = []
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if split != "all" and row["split"] != split:
                    continue
                image_path = self.root / row["image"]
                mask_path = self.root / row["mask"] if row.get("mask") else None
                if not image_path.exists():
                    raise FileNotFoundError(f"VisA image referenced by CSV is missing: {image_path}")
                if mask_path is not None and not mask_path.exists():
                    raise FileNotFoundError(f"VisA mask referenced by CSV is missing: {mask_path}")
                self.samples.append({
                    "image": image_path,
                    "mask": mask_path,
                    "object": row["object"],
                    "label": row["label"],
                    "id": image_path.relative_to(self.root).as_posix(),
                })
        if not self.samples:
            raise RuntimeError(f"No VisA samples found for split={split}, split_csv={split_csv}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        item = self.samples[index]
        image_array = cv2.imread(str(item["image"]), cv2.IMREAD_COLOR)
        if image_array is None:
            raise ValueError(f"Cannot read VisA image: {item['image']}")
        image = torch.from_numpy(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        height, width = image.shape[-2:]
        mask = torch.zeros((height, width), dtype=torch.long)
        if item["mask"] is not None:
            mask_array = cv2.imread(str(item["mask"]), cv2.IMREAD_GRAYSCALE)
            if mask_array is None:
                raise ValueError(f"Cannot read VisA mask: {item['mask']}")
            mask = (torch.from_numpy(mask_array) > 0).long()
        if self.image_size is not None:
            size = (self.image_size, self.image_size) if isinstance(self.image_size, int) else self.image_size
            image = F.interpolate(image.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)
            mask = F.interpolate(mask[None, None].float(), size=size, mode="nearest").squeeze(0).squeeze(0).long()
        sample = {
            "image": image,
            "masks": mask.unsqueeze(0),
            "mask": mask,
            "image_id": item["id"],
            "id": item["id"],
            "object": item["object"],
            "label": item["label"],
            "anomaly": int(item["label"] == "anomaly"),
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
            sample["mask"] = sample["masks"].squeeze(0).long()
        return sample
