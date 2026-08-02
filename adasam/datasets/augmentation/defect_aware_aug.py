"""Mask-aware copy-paste and scale perturbation for industrial defects."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from adasam.datasets.augmentation.basic_aug import BasicAugmentation


class DefectAwareAugmentation(BasicAugmentation):
    """Basic augmentation followed by bounded same-image defect copy-paste."""

    def __init__(
        self,
        copy_paste_probability: float = 0.7,
        scale_range: tuple[float, float] = (0.65, 1.25),
        max_added_area_ratio: float = 0.15,
        placement_attempts: int = 12,
        **basic_options,
    ) -> None:
        super().__init__(**basic_options)
        if not 0.0 < scale_range[0] <= scale_range[1]:
            raise ValueError("scale_range must be positive and ordered")
        if not 0.0 < max_added_area_ratio < 1.0:
            raise ValueError("max_added_area_ratio must be in (0,1)")
        self.copy_paste_probability = copy_paste_probability
        self.scale_range = scale_range
        self.max_added_area_ratio = max_added_area_ratio
        self.placement_attempts = placement_attempts

    def __call__(self, sample: dict) -> dict:
        augmented = super().__call__(sample)
        if torch.rand(()) >= self.copy_paste_probability:
            return augmented
        image, mask = augmented["image"].clone(), augmented["masks"].clone()
        class_ids = torch.unique(mask)
        class_ids = class_ids[class_ids > 0]
        if class_ids.numel() == 0:
            return augmented
        class_id = int(class_ids[torch.randint(class_ids.numel(), ())].item())
        selected = mask[0] == class_id
        coordinates = selected.nonzero(as_tuple=False)
        y0, x0 = coordinates.min(dim=0).values.tolist()
        y1, x1 = (coordinates.max(dim=0).values + 1).tolist()
        source_image = image[:, y0:y1, x0:x1]
        source_mask = selected[y0:y1, x0:x1]

        scale = float(torch.empty(()).uniform_(*self.scale_range))
        source_h, source_w = source_mask.shape
        target_h = max(1, round(source_h * scale))
        target_w = max(1, round(source_w * scale))
        image_h, image_w = mask.shape[-2:]
        area_limit = max(1, round(image_h * image_w * self.max_added_area_ratio))
        source_pixels = max(1, int(source_mask.sum()))
        if source_pixels * scale * scale > area_limit:
            scale = (area_limit / source_pixels) ** 0.5
            target_h = max(1, round(source_h * scale))
            target_w = max(1, round(source_w * scale))
        target_h, target_w = min(target_h, image_h), min(target_w, image_w)
        patch = F.interpolate(source_image.unsqueeze(0), (target_h, target_w), mode="bilinear", align_corners=False).squeeze(0)
        patch_mask = F.interpolate(
            source_mask[None, None].float(), (target_h, target_w), mode="nearest"
        ).squeeze(0).squeeze(0).bool()
        if not patch_mask.any():
            return augmented

        for _ in range(self.placement_attempts):
            paste_y = int(torch.randint(0, image_h - target_h + 1, ()).item())
            paste_x = int(torch.randint(0, image_w - target_w + 1, ()).item())
            destination_mask = mask[0, paste_y:paste_y + target_h, paste_x:paste_x + target_w]
            writable = patch_mask & (destination_mask == 0)
            if int(writable.sum()) < max(1, round(float(patch_mask.sum()) * 0.8)):
                continue
            destination_image = image[:, paste_y:paste_y + target_h, paste_x:paste_x + target_w]
            destination_image[:, writable] = patch[:, writable]
            destination_mask[writable] = class_id
            break
        augmented["image"] = image.contiguous()
        augmented["masks"] = mask.contiguous()
        return augmented
