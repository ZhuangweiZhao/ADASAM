"""
几何先验模块 | Geometric Prior Module.
========================================

基于 support-query 余弦相似度生成几何空间先验, 与 Semantic Prior Generator
形成双支路架构。

Generates a geometric spatial prior from support-query cosine similarity,
forming a dual-branch architecture with the Semantic Prior Generator.

两条支路 | Two branches:
    - Geometric Prior: support-query 相似性 → "目标大概在哪里" (几何先验)
    - Semantic Prior: SPG 学习到的 → "目标是什么" (语义先验)

    - Geometric Prior: support-query similarity → "where" (geometric)
    - Semantic Prior: learned SPG output → "what" (semantic)

参考 | Reference:
    - SAM-RSP: "Representation Prompting for SAM-based Few-shot Segmentation"
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricPriorModule(nn.Module):
    """几何先验生成器: support-query 相似度 + 像素级查询原型.

    Geometric prior generator: support-query similarity (RSP) + pixel-level
    query prototype via self-correlation gating.

    :param embed_dim: 特征维度 (SAM token dim, 256).
    """

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        C = embed_dim

        # ── RSP projection layers ──
        self.query_rsp_proj = nn.Conv2d(C, C, kernel_size=1, bias=False)
        self.support_rsp_proj = nn.Linear(C, C, bias=False)

        # ── Merge [query_features, rsp_map] → geometric prior ──
        self.merge = nn.Sequential(
            nn.Conv2d(C + 1, C, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        query_features: torch.Tensor,
        support_memory: torch.Tensor,
    ) -> torch.Tensor:
        """生成几何先验 | Generate geometric prior.

        :param query_features: [1, C, gh, gw] query image features.
        :param support_memory: [M, C] support memory tokens.
        :return: geometric_prior [1, C, gh, gw].
        """
        B, C, H, W = query_features.shape
        N = H * W
        has_support = support_memory.shape[0] > 0

        # ═════════════════════════════════════════════════════
        # RSP: support-query cosine similarity → spatial prior
        # ═════════════════════════════════════════════════════
        if has_support:
            q_rsp = self.query_rsp_proj(query_features)          # [B, C, H, W]
            s_rsp = self.support_rsp_proj(support_memory)        # [M, C]

            q_norm = F.normalize(q_rsp.reshape(B, C, N), dim=1)  # [B, C, N]
            s_norm = F.normalize(s_rsp, dim=1)                    # [M, C]

            sim = torch.einsum("bcn,mc->bmn", q_norm, s_norm)    # [B, M, N]
            sim = sim.max(dim=1)[0]                                # [B, N]
            rsp_map = sim.reshape(B, 1, H, W)                     # [B, 1, H, W]

            rsp_min = rsp_map.amin(dim=(2, 3), keepdim=True)
            rsp_max = rsp_map.amax(dim=(2, 3), keepdim=True)
            rsp_map = (rsp_map - rsp_min) / (rsp_max - rsp_min + 1e-5)
        else:
            rsp_map = torch.zeros(B, 1, H, W, device=query_features.device)

        # ═════════════════════════════════════════════════════
        # Merge: query features × rsp map → geometric prior
        # ═════════════════════════════════════════════════════
        # 与旧 CoarsePrior 的区别: 不再输出复杂的 pixel_prototype,
        # 而是将 RSP map 和 query features 融合为统一的 geometric prior。
        # Difference from old CoarsePrior: no pixel_prototype output;
        # RSP map + query features fused into unified geometric prior.
        geometric_prior = self.merge(
            torch.cat([query_features, rsp_map], dim=1)
        )  # [1, C, H, W]

        return geometric_prior
