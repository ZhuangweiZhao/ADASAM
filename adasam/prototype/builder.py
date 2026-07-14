"""
原型构建 | Prototype Builder.
=============================

从 K 张 support (图像嵌入 + 前景掩码) 构建一个类原型 (256-d 向量)。
Build one class prototype (256-d vector) from K support (image embedding + FG mask) pairs.

方法 | Method: 前景掩码平均池化 (masked average pooling) —— 每张 support 在其掩码前景区域内
对嵌入做平均, L2 归一化; 再对 K 张求平均并再次归一化。此即 AdaTile-FastSAM
``compute_fg_prototype`` 的规范定义, 但工作在 MobileSAM 的 256-d 嵌入 (非 FastSAM 1280-d P4)。
Per-support masked average pooling over the FG region, L2-normalized; then averaged over K
supports and re-normalized. Same canonical definition as compute_fg_prototype, but on the
MobileSAM 256-d embedding (not FastSAM's 1280-d P4).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from adasam.utils.transforms import resize_mask


class PrototypeBuilder:
    """前景掩码平均池化的原型构建器 | Masked-average-pooling prototype builder.

    :param embed_dim: 嵌入/原型维度 | embedding & prototype dim (MobileSAM = 256).
    """

    def __init__(self, embed_dim: int = 256) -> None:
        self.embed_dim = embed_dim

    def build(
        self,
        support_embeddings: list[torch.Tensor],
        support_masks: list[torch.Tensor],
    ) -> torch.Tensor:
        """构建类原型 | Build a class prototype.

        :param support_embeddings: K 个 [C, gh, gw] 图像嵌入 | K image embeddings.
        :param support_masks: K 个 [H, W] 前景掩码 (bool/float) | K foreground masks.
        :return: [embed_dim] L2-归一化原型; 全空掩码时返回零向量 | L2-normalized prototype;
            zero vector if every support mask is empty.
        """
        if len(support_embeddings) != len(support_masks):
            raise ValueError("support_embeddings and support_masks must have equal length")

        device = support_embeddings[0].device if support_embeddings else torch.device("cpu")

        per_support: list[torch.Tensor] = []
        for emb, mask in zip(support_embeddings, support_masks):
            if emb.ndim != 3:
                raise ValueError(f"expected embedding [C, gh, gw], got {tuple(emb.shape)}")
            c, gh, gw = emb.shape
            m = resize_mask(mask, (gh, gw)).to(device)          # [gh, gw] ∈ {0,1}
            denom = m.sum()
            if denom < 1.0:
                continue                                        # 空掩码跳过 | skip empty mask
            pooled = (emb * m.unsqueeze(0)).flatten(1).sum(dim=1) / denom   # [C]
            per_support.append(F.normalize(pooled, dim=0))

        if not per_support:
            return torch.zeros(self.embed_dim, device=device)

        proto = torch.stack(per_support, dim=0).mean(dim=0)     # [C]
        return F.normalize(proto, dim=0)
