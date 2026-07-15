"""
CAT-SAM Feature Adapter | CAT-SAM 特征适配器.
==============================================

CAT-SAM 论文启发的轻量瓶颈残差适配器, 用于 MobileSAM 特征域适配。
A CAT-SAM-inspired lightweight bottleneck residual adapter for MobileSAM
feature domain adaptation.

设计 | Design:
    - 放置在冻结 MobileSAM 编码器之后, 对 256-d 特征做域适配。
      Placed after the frozen MobileSAM encoder to adapt 256-d features.
    - 瓶颈结构: 256 → 64 → 64 (3×3 conv) → 256, 参数量 ~70K。
      Bottleneck: 256 → 64 → 64 (3×3 conv) → 256, ~70K params.
    - 残差连接 + 末层零初始化 → 训练起点等价于无适配器。
      Residual connection + zero-init final layer → training starts as identity.
    - 支持/查询图像共享同一适配器 (Siamese).
      Support and query images share the same adapter (Siamese).

与 CAT-SAM 论文的关系 | Relationship to CAT-SAM paper:
    CAT-SAM-A 在 ViT 每个 block 前注入 adapter 特征; MobileSAM 使用 TinyViT,
    内部结构与 ViT 不同, 因此改用编码器后置适配器, 精神一致。
    CAT-SAM-A injects adapter features before each ViT block; MobileSAM uses
    TinyViT whose internals differ from ViT, so we use a post-encoder adapter
    which is in the same spirit.

参考 | Reference:
    Xiao et al., "CAT-SAM: Conditional Tuning for Few-Shot Adaptation of
    Segment Anything Model", ECCV 2024.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CATAdapter(nn.Module):
    """CAT-SAM 启发的特征适配器 | CAT-SAM-inspired feature adapter.

    对冻结编码器的输出做轻量域适配, 保留特征维度不变。
    Lightweight domain adaptation on frozen encoder output; preserves feature dim.

    :param dim: 特征维度 (256 for MobileSAM) | feature dimension.
    :param bottleneck: 瓶颈维度 | bottleneck dimension.
    """

    def __init__(self, dim: int = 256, bottleneck: int = 64) -> None:
        super().__init__()
        self.dim = dim
        self.bottleneck = bottleneck

        # 下投影 | down-project
        self.down = nn.Conv2d(dim, bottleneck, kernel_size=1, bias=False)

        # 空间上下文 | spatial context
        self.spatial = nn.Conv2d(
            bottleneck, bottleneck, kernel_size=3, padding=1, bias=False
        )

        # 上投影 (零初始化 → 训练起点为恒等) | up-project (zero-init → identity start)
        self.up = nn.Conv2d(bottleneck, dim, kernel_size=1, bias=False)

        # 零初始化末层 | zero-init final layer
        nn.init.zeros_(self.up.weight)

        # 统计 | stats
        self._n_params = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 | Forward pass.

        :param x: [B, C, H, W] 编码器输出 | encoder output.
        :return: [B, C, H, W] 适配后特征 | adapted features.
        """
        identity = x
        out = self.down(x)
        out = F.gelu(out)
        out = self.spatial(out)
        out = F.gelu(out)
        out = self.up(out)
        return identity + out

    def extra_repr(self) -> str:
        return f"dim={self.dim}, bottleneck={self.bottleneck}, params={self._n_params:,}"
