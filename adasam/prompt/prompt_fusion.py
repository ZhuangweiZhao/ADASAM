"""
双支路先验融合 | Dual-Prior Prompt Fusion.
============================================

将 Geometric Prior 和 Semantic Prior 融合为 SAM Decoder 所需的
dense prompt + sparse token。

Fuses Geometric Prior and Semantic Prior into the dense prompt +
sparse token required by the SAM Decoder.

融合模式 | Fusion modes:
    - "concat": 拼接 → 1×1 Conv 降维 (默认, 最灵活)
    - "add": 逐元素相加 (参数最少)
    - "gated": 可学习门控融合 (GP × gate + SP × (1-gate))
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PromptFusion(nn.Module):
    """双支路先验融合模块 | Dual-prior prompt fusion module.

    :param embed_dim: 特征维度 | feature dimension (256).
    :param mode: 融合模式 | fusion mode: "concat" | "add" | "gated".
    """

    def __init__(self, embed_dim: int = 256, mode: str = "concat") -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.mode = mode
        C = embed_dim

        if mode == "concat":
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(C * 2, C, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False),
            )
        elif mode == "gated":
            self.gate_conv = nn.Sequential(
                nn.Conv2d(C * 2, 1, kernel_size=1),
                nn.Sigmoid(),
            )
        elif mode == "add":
            # No extra params — just element-wise addition
            pass
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")

    def forward(
        self,
        geometric_prior: torch.Tensor,
        semantic_prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """融合双先验 → dense prompt + sparse token.

        :param geometric_prior: [1, C, H, W] from GeometricPriorModule.
        :param semantic_prior: [1, C, H, W] from SemanticPriorGenerator.
        :return: (dense_prompt [1, C, H, W], sparse_token [1, C]).
        """
        if self.mode == "concat":
            fused = self.fusion_conv(
                torch.cat([geometric_prior, semantic_prior], dim=1)
            )
        elif self.mode == "gated":
            gate = self.gate_conv(
                torch.cat([geometric_prior, semantic_prior], dim=1)
            )  # [1, 1, H, W]
            fused = gate * geometric_prior + (1 - gate) * semantic_prior
        elif self.mode == "add":
            fused = geometric_prior + semantic_prior

        dense_prompt = fused  # [1, C, H, W]

        # sparse_token: spatial mean pool → single conditioning token
        sparse_token = fused.mean(dim=(2, 3))  # [1, C]

        return dense_prompt, sparse_token
