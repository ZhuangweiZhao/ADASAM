"""
SAM-RSP 共享 transforms 与工具函数 | Shared transforms &amp; utilities for SAM-RSP stages.
=======================================================================================

原本 sam_rsp_stage{1,2,3}.py 各自重复定义这些类, 提取到此统一维护。
Originally duplicated across sam_rsp_stage{1,2,3}.py; extracted here for single-source maintenance.

使用 | Usage::

    from tools.sam_rsp.common import (
        Compose, RandomScale, RandomCrop, RandomHorizontalFlip,
        Resize, Normalize, ToTensor, EpisodeTransformWrapper,
        poly_lr, compute_fb_iou, IMAGENET_MEAN, IMAGENET_STD,
    )
"""

from __future__ import annotations

import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0


# ═══════════════════════════════════════════════════════════════════
# Transforms
# ═══════════════════════════════════════════════════════════════════

class Compose:
    """顺序执行多个 transform."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class RandomScale:
    """随机缩放 (0.5x – 2.0x)."""

    def __init__(self, scale_range=(0.5, 2.0)):
        self.scale_range = scale_range

    def __call__(self, image, mask):
        import cv2
        scale = np.random.uniform(*self.scale_range)
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return image, mask


class RandomCrop:
    """随机裁剪至固定尺寸."""

    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, image, mask):
        import cv2
        h, w = image.shape[:2]
        th, tw = self.size
        if h < th or w < tw:
            image = cv2.resize(image, (max(w, tw), max(h, th)), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (max(w, tw), max(h, th)), interpolation=cv2.INTER_NEAREST)
            h, w = image.shape[:2]
        y = np.random.randint(0, h - th + 1)
        x = np.random.randint(0, w - tw + 1)
        return image[y:y + th, x:x + tw], mask[y:y + th, x:x + tw]


class RandomHorizontalFlip:
    """随机水平翻转."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if np.random.random() < self.p:
            image = np.ascontiguousarray(np.fliplr(image))
            mask = np.ascontiguousarray(np.fliplr(mask))
        return image, mask


class Resize:
    """强制缩放至固定尺寸."""

    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, image, mask):
        import cv2
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        return image, mask


class Normalize:
    """ImageNet 归一化."""

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        image = (image - self.mean) / self.std
        return image, mask


class ToTensor:
    """无操作占位 transform (Stage 1 用)."""
    def __call__(self, image, mask):
        return image, mask


# ═══════════════════════════════════════════════════════════════════
# Episode Transform Wrapper
# ═══════════════════════════════════════════════════════════════════

class EpisodeTransformWrapper:
    """对 query 和各 support image/mask 独立应用 transform."""

    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        q_image = item["query_image"].numpy().transpose(1, 2, 0)
        q_mask = item["query_mask"].numpy()

        q_image, q_mask = self.transform(q_image, q_mask)

        s_images = item["support_images"].numpy()
        s_masks = item["support_masks"].numpy()
        K = s_images.shape[0]
        s_imgs_out, s_msks_out = [], []
        for k in range(K):
            s_img_k = s_images[k].transpose(1, 2, 0)
            s_msk_k = s_masks[k]
            si, sm = self.transform(s_img_k, s_msk_k)
            s_imgs_out.append(torch.from_numpy(si).permute(2, 0, 1))
            s_msks_out.append(torch.from_numpy(sm))

        return {
            "query_image": torch.from_numpy(q_image).permute(2, 0, 1).float(),
            "query_mask": torch.from_numpy(q_mask).long(),
            "support_images": torch.stack(s_imgs_out, 0).float(),
            "support_masks": torch.stack(s_msks_out, 0).long(),
            "class_id": item["class_id"],
            "subcls": item["subcls"],
        }


# ═══════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════

def poly_lr(optimizer, base_lr, current_iter, max_iter, power=0.9,
            warmup=False, warmup_step=0):
    """Poly learning rate schedule (同 SAM-RSP 原版)."""
    if warmup and current_iter < warmup_step:
        lr = base_lr * (current_iter + 1) / warmup_step
    else:
        lr = base_lr * (1 - float(current_iter) / float(max_iter)) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_fb_iou(pred: np.ndarray, target: np.ndarray) -> dict:
    """Foreground-Background IoU.

    Args:
        pred: [H, W] binary prediction (1=FG, 0=BG)
        target: [H, W] binary ground truth
    Returns:
        {"FG-IoU": ..., "BG-IoU": ..., "FB-IoU": ...}
    """
    fg_pred = pred == 1
    fg_gt = target == 1
    bg_pred = ~fg_pred
    bg_gt = ~fg_gt

    fg_inter = (fg_pred & fg_gt).sum()
    fg_union = (fg_pred | fg_gt).sum()
    bg_inter = (bg_pred & bg_gt).sum()
    bg_union = (bg_pred | bg_gt).sum()

    fg_iou = fg_inter / max(fg_union, 1)
    bg_iou = bg_inter / max(bg_union, 1)
    return {"FG-IoU": fg_iou, "BG-IoU": bg_iou, "FB-IoU": (fg_iou + bg_iou) / 2}
