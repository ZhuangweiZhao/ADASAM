"""Interpretable geometric and photometric segmentation augmentation."""

from __future__ import annotations

import torch


class BasicAugmentation:
    """Apply synchronized flips/rotations and image-only intensity jitter."""

    def __init__(
        self,
        horizontal_flip_probability: float = 0.5,
        vertical_flip_probability: float = 0.5,
        rotation_probability: float = 0.5,
        brightness: float = 0.15,
        contrast: float = 0.15,
    ) -> None:
        self.horizontal_flip_probability = horizontal_flip_probability
        self.vertical_flip_probability = vertical_flip_probability
        self.rotation_probability = rotation_probability
        self.brightness = brightness
        self.contrast = contrast

    @staticmethod
    def _validate(sample: dict) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = sample["image"], sample["masks"]
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("sample image must have shape [3,H,W]")
        if mask.ndim != 3 or mask.shape[0] != 1 or mask.shape[-2:] != image.shape[-2:]:
            raise ValueError("sample masks must have shape [1,H,W] matching image")
        return image, mask

    def __call__(self, sample: dict) -> dict:
        image, mask = self._validate(sample)
        image, mask = image.clone(), mask.clone()
        if torch.rand(()) < self.horizontal_flip_probability:
            image, mask = image.flip(-1), mask.flip(-1)
        if torch.rand(()) < self.vertical_flip_probability:
            image, mask = image.flip(-2), mask.flip(-2)
        if torch.rand(()) < self.rotation_probability:
            turns = int(torch.randint(1, 4, ()).item())
            image, mask = torch.rot90(image, turns, (-2, -1)), torch.rot90(mask, turns, (-2, -1))
        if self.brightness > 0:
            factor = 1.0 + float(torch.empty(()).uniform_(-self.brightness, self.brightness))
            image = image * factor
        if self.contrast > 0:
            factor = 1.0 + float(torch.empty(()).uniform_(-self.contrast, self.contrast))
            mean = image.mean(dim=(-2, -1), keepdim=True)
            image = (image - mean) * factor + mean
        augmented = dict(sample)
        augmented["image"] = image.clamp(0.0, 1.0).contiguous()
        augmented["masks"] = mask.contiguous()
        augmented["image_size"] = tuple(image.shape[-2:])
        return augmented
