"""Industrial datasets for standard, non-episodic semantic segmentation."""

from adasam.datasets.industrial.base import LabelRatioSubset, fixed_validation_split_indices
from adasam.datasets.industrial.neu_seg import NEUSegSemanticDataset
from adasam.datasets.industrial.loveda import LoveDASemanticDataset

__all__ = [
    "LabelRatioSubset",
    "NEUSegSemanticDataset",
    "LoveDASemanticDataset",
    "fixed_validation_split_indices",
]
