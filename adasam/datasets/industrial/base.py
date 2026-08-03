"""Common contracts for standard industrial semantic segmentation datasets."""

from __future__ import annotations

import random
from collections.abc import Sequence

from torch.utils.data import Dataset


VALID_LABEL_RATIOS = {1, 5, 10, 20, 25, 50, 100}


def fixed_validation_split_indices(
    dataset_size: int,
    ratio: int,
    seed: int = 42,
    validation_fraction: float = 0.2,
    validation_seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Return nested train indices and a validation set fixed across ratios/seeds."""
    if ratio not in VALID_LABEL_RATIOS:
        raise ValueError(f"label_ratio must be one of {sorted(VALID_LABEL_RATIOS)}")
    if dataset_size < 2:
        raise ValueError("dataset must contain at least two samples")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    indices = list(range(dataset_size))
    random.Random(validation_seed).shuffle(indices)
    validation_count = max(1, round(dataset_size * validation_fraction))
    if validation_count >= dataset_size:
        raise ValueError("validation split leaves no training samples")
    validation_indices = sorted(indices[:validation_count])
    training_pool = indices[validation_count:]
    random.Random(seed).shuffle(training_pool)
    labeled_count = (
        len(training_pool)
        if ratio == 100
        else max(1, round(len(training_pool) * ratio / 100))
    )
    return training_pool[:labeled_count], validation_indices, training_pool


class LabelRatioSubset(Dataset):
    """Deterministic nested percentage subset of a semantic dataset."""

    def __init__(self, dataset: Dataset, ratio: int, seed: int = 42) -> None:
        if ratio not in VALID_LABEL_RATIOS:
            raise ValueError(f"label_ratio must be one of {sorted(VALID_LABEL_RATIOS)}")
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
