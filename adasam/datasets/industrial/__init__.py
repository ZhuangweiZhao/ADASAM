"""Industrial datasets for standard, non-episodic semantic segmentation."""

from adasam.datasets.industrial.base import LabelRatioSubset
from adasam.datasets.industrial.neu_seg import NEUSegSemanticDataset

__all__ = ["LabelRatioSubset", "NEUSegSemanticDataset"]
