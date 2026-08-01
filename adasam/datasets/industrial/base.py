"""Common contracts for standard industrial semantic segmentation datasets."""

from __future__ import annotations

import random
from collections.abc import Sequence

from torch.utils.data import Dataset


class LabelRatioSubset(Dataset):
    """Deterministic nested percentage subset of a semantic dataset."""

    def __init__(self, dataset: Dataset, ratio: int, seed: int = 42) -> None:
        if ratio not in {1, 5, 10, 25, 100}:
            raise ValueError("label_ratio must be one of 1, 5, 10, 25, 100")
        self.dataset = dataset
        indices = list(range(len(dataset)))
        random.Random(seed).shuffle(indices)
        count = len(indices) if ratio == 100 else max(1, round(len(indices) * ratio / 100))
        self.indices: Sequence[int] = indices[:count]
        self.ratio = ratio

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        return self.dataset[self.indices[index]]
