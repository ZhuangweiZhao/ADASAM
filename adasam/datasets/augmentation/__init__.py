"""Synchronized image/mask augmentation for industrial segmentation."""

from adasam.datasets.augmentation.basic_aug import BasicAugmentation, RemoteSensingStrongAugmentation
from adasam.datasets.augmentation.defect_aware_aug import DefectAwareAugmentation


def build_augmentation(mode: str):
    """Build a training transform; ``none`` intentionally returns no transform."""
    if mode == "none":
        return None
    if mode == "basic":
        return BasicAugmentation()
    if mode == "remote_strong":
        return RemoteSensingStrongAugmentation()
    if mode == "defect":
        return DefectAwareAugmentation()
    raise ValueError("augmentation mode must be one of: none, basic, remote_strong, defect")


__all__ = [
    "BasicAugmentation", "RemoteSensingStrongAugmentation",
    "DefectAwareAugmentation", "build_augmentation",
]
