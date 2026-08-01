"""Standard semantic segmentation view of the NEU_Seg dataset."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from adasam.datasets.neu_seg import NEUSegDataset


class NEUSegSemanticDataset(Dataset):
    """Return image, dense class mask and sample ID without episode semantics."""

    NUM_CLASSES = NEUSegDataset.NUM_CLASSES
    CLASS_NAMES = NEUSegDataset.CLASS_NAMES

    def __init__(self, root: str | Path, split: str = "train", transforms=None) -> None:
        self.dataset = NEUSegDataset(root, split=split, transforms=transforms)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index]
        return {
            "image": sample["image"],
            "mask": sample["masks"].squeeze(0).long(),
            "id": sample["image_id"],
        }
