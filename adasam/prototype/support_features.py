"""
Dense Support Features — thin wrapper | 密集支持特征 — 薄封装.
================================================================

从 K 张 support 图像中同时提取:
  - Dense Support Features [K, 256, 64, 64] (每张 support 的完整空间嵌入)
  - Global Prototype [256] (PrototypeBuilder 输出)

这替代了之前 Trainer._build_prototype() 中"提取嵌入 → 丢弃, 只留原型"的模式。
现在保留每张 support 的完整嵌入, 供 Correlation Builder 使用。

Extract from K support images simultaneously:
  - Dense Support Features [K, 256, 64, 64] (full spatial embedding per support)
  - Global Prototype [256] (PrototypeBuilder output)

This replaces the prior pattern in Trainer._build_prototype() where embeddings
were extracted then discarded, keeping only the prototype. Now we keep the full
per-support embeddings for the Correlation Builder downstream.
"""

from __future__ import annotations

import torch

from adasam.prototype.builder import PrototypeBuilder


def extract_support_features(
    backbone,                           # MobileSAMBackbone (frozen)
    support_images: list[torch.Tensor],  # K × [3, 896, 896] float in [0, 1]
    support_fg_masks: list[torch.Tensor], # K × [896, 896] float in {0, 1}
    proto_builder: PrototypeBuilder | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """提取密集支持特征与全局原型 | Extract dense support features & global prototype.

    :param backbone: MobileSAMBackbone instance (frozen, eval mode).
    :param support_images: K preprocessed images [3, H, W] float in [0,1].
    :param support_fg_masks: K FG masks [H, W] float.
    :param proto_builder: PrototypeBuilder instance (auto-created if None).
    :return: (support_features [K, 256, 64, 64], prototype [256]).
    """
    if len(support_images) != len(support_fg_masks):
        raise ValueError(
            f"Mismatch: {len(support_images)} images vs {len(support_fg_masks)} masks"
        )

    if proto_builder is None:
        proto_builder = PrototypeBuilder(embed_dim=256)

    embeddings: list[torch.Tensor] = []
    masks_at_grid: list[torch.Tensor] = []

    for img, fg_mask in zip(support_images, support_fg_masks):
        # Run backbone on single image — returns {"image_embedding": [1, 256, 64, 64]}
        # The backbone expects [B, 3, 1024, 1024] normalized input.
        # We receive images already preprocessed (or raw at tile resolution).
        emb = backbone(img.unsqueeze(0))["image_embedding"]  # [1, 256, 64, 64]
        embeddings.append(emb[0])                            # [256, 64, 64]
        masks_at_grid.append(fg_mask)

    # Build global prototype (same as before)
    prototype = proto_builder.build(embeddings, masks_at_grid)  # [256]

    # Stack into a single tensor
    support_features = torch.stack(embeddings, dim=0)  # [K, 256, 64, 64]

    return support_features, prototype
