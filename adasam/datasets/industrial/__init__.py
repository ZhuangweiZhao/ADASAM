"""Industrial datasets for standard, non-episodic semantic segmentation."""

from adasam.datasets.industrial.base import LabelRatioSubset, fixed_validation_split_indices
from adasam.datasets.industrial.neu_seg import NEUSegSemanticDataset
from adasam.datasets.industrial.loveda import LoveDASemanticDataset
from adasam.datasets.industrial.visa import VisASemanticDataset
from adasam.datasets.industrial.isaid import ISAIDSemanticDataset
from adasam.datasets.industrial.vaihingen import VaihingenSemanticDataset

__all__ = [
    "LabelRatioSubset",
    "NEUSegSemanticDataset",
    "LoveDASemanticDataset",
    "VisASemanticDataset",
    "ISAIDSemanticDataset",
    "VaihingenSemanticDataset",
    "fixed_validation_split_indices",
]
