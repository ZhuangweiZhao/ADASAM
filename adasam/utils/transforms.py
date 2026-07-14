"""
SAM 图像预处理 | SAM image preprocessing.
==========================================

将任意分辨率的 RGB 图像转换为 MobileSAM 图像编码器所需的规范输入:
Convert an arbitrary-resolution RGB image into the canonical input expected by the
MobileSAM image encoder:

    resize-longest-side → 1024  ·  normalize (pixel_mean/std, 0-255)  ·  pad → 1024×1024

与 mobile_sam.modeling.sam.Sam.preprocess 语义一致 (常量相同), 但从 backbone 中解耦,
使 backbone.forward 只接收已预处理张量 [B,3,1024,1024] —— 单一职责。
Semantics identical to Sam.preprocess (same constants) but decoupled from the backbone,
so backbone.forward only ever receives a preprocessed [B,3,1024,1024] tensor (SRP).

遥感 tile 为 896² 方形 → resize 到 1024² (无 padding); GT/评估仍在 896² 进行。
Aerial tiles are square 896² → resize to 1024² (no padding); GT/eval stay at 896².
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# ── SAM 归一化常量 (0-255 尺度) | SAM normalization constants (0-255 scale) ──
PIXEL_MEAN: tuple[float, float, float] = (123.675, 116.28, 103.53)
PIXEL_STD: tuple[float, float, float] = (58.395, 57.12, 57.375)
SAM_IMAGE_SIZE: int = 1024


@dataclass(frozen=True)
class PreprocessMeta:
    """预处理元数据, 供 postprocess 将掩码映射回原图 | Metadata to map masks back to the original image.

    :param original_size: resize 前的原始尺寸 (H, W) | original size before resize.
    :param input_size: resize 后、pad 前的尺寸 (H, W) | size after resize, before padding.
    """

    original_size: tuple[int, int]
    input_size: tuple[int, int]


def resize_longest_side(h: int, w: int, target: int = SAM_IMAGE_SIZE) -> tuple[int, int]:
    """按最长边缩放到 target, 保持长宽比 | Scale so the longest side equals target, keep aspect ratio.

    :return: 缩放后的 (h, w) | resized (h, w).
    """
    scale = target / max(h, w)
    return int(round(h * scale)), int(round(w * scale))


def _to_chw_float(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    """统一转为 [3, H, W] float32 (0-255 尺度) 张量 | Normalize input to a [3,H,W] float32 (0-255) tensor.

    接受 | Accepts:
        - np.ndarray [H, W, 3] uint8/float (RGB)
        - torch.Tensor [3, H, W] 或 [H, W, 3] (uint8 或 float; float∈[0,1] 会被放大到 0-255)
    """
    if isinstance(image, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(image))
    else:
        t = image
    t = t.detach().float()

    # HWC → CHW
    if t.ndim == 3 and t.shape[-1] == 3 and t.shape[0] != 3:
        t = t.permute(2, 0, 1).contiguous()
    if t.ndim != 3 or t.shape[0] != 3:
        raise ValueError(f"expected a 3-channel image, got shape {tuple(t.shape)}")

    # 若像素在 [0,1] (float 归一化输入), 放大到 0-255 以匹配 SAM 常量
    # If pixels are in [0,1] (float normalized input), scale to 0-255 to match SAM constants.
    if t.max() <= 1.0 + 1e-6:
        t = t * 255.0
    return t


def preprocess_image(
    image: np.ndarray | torch.Tensor,
    sam_image_size: int = SAM_IMAGE_SIZE,
    pixel_mean: tuple[float, float, float] = PIXEL_MEAN,
    pixel_std: tuple[float, float, float] = PIXEL_STD,
) -> tuple[torch.Tensor, PreprocessMeta]:
    """RGB 图像 → MobileSAM 编码器输入 | RGB image → MobileSAM encoder input.

    :param image: [H, W, 3] uint8 RGB (numpy) 或 [3, H, W]/[H, W, 3] 张量 | RGB image.
    :return: (tensor[3, sam_image_size, sam_image_size] float32, meta). 未加 batch 维,
        由调用方 stack 成 [B, 3, S, S] | unbatched CHW tensor; caller stacks into a batch.
    """
    x = _to_chw_float(image)                      # [3, H, W], 0-255
    orig_h, orig_w = int(x.shape[1]), int(x.shape[2])

    new_h, new_w = resize_longest_side(orig_h, orig_w, sam_image_size)
    x = F.interpolate(
        x.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    ).squeeze(0)                                  # [3, new_h, new_w]

    mean = torch.tensor(pixel_mean, dtype=x.dtype).view(3, 1, 1)
    std = torch.tensor(pixel_std, dtype=x.dtype).view(3, 1, 1)
    x = (x - mean) / std

    pad_h = sam_image_size - new_h
    pad_w = sam_image_size - new_w
    x = F.pad(x, (0, pad_w, 0, pad_h))            # 右/下补零 | pad right/bottom
    return x, PreprocessMeta(original_size=(orig_h, orig_w), input_size=(new_h, new_w))


def resize_mask(
    mask: np.ndarray | torch.Tensor, size: int | tuple[int, int]
) -> torch.Tensor:
    """最近邻缩放二值掩码 | Nearest-neighbor resize of a binary mask.

    用于把支持集掩码降到编码器网格 (64²) 或把预测掩码对齐到 tile 分辨率。
    Used to downscale support masks to the encoder grid (64²) or align predicted masks.

    :param mask: [H, W] bool/uint8/float 掩码 | binary mask.
    :param size: 目标边长 int 或 (H, W) | target side (int) or (H, W).
    :return: [h, w] float32 张量 (0/1) | float32 tensor in {0,1}.
    """
    if isinstance(mask, np.ndarray):
        m = torch.from_numpy(np.ascontiguousarray(mask))
    else:
        m = mask
    m = m.detach().float()
    if m.ndim != 2:
        raise ValueError(f"expected a [H, W] mask, got shape {tuple(m.shape)}")

    out_size = (size, size) if isinstance(size, int) else size
    m = F.interpolate(m[None, None], size=out_size, mode="nearest")[0, 0]
    return m
