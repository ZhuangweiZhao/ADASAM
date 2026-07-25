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

        # Identity init: at initialization, cosine similarity is computed
        # in the raw feature space (which we verified has strong FG/BG signal).
        # Projection layers then learn to refine which dimensions matter during training.
        with torch.no_grad():
            w_q = self.query_rsp_proj.weight
            w_s = self.support_rsp_proj.weight
            w_q.zero_()
            w_s.zero_()
            for i in range(min(C, C)):
                w_q[i, i, 0, 0] = 1.0
                w_s[i, i] = 1.0

        # ── Merge: rsp_map → geometric prior ──
        # Simple 1×1 Conv to expand rsp_map [1, H, W] → [1, C, H, W].
        # Identity-initialized so it passes the cosine signal through unchanged at init;
        # training can learn channel-wise re-weighting.
        self.rsp_expand = nn.Conv2d(1, C, kernel_size=1, bias=False)
        with torch.no_grad():
            w_e = self.rsp_expand.weight
            w_e.zero_()
            for i in range(min(C, C)):
                w_e[i, 0, 0, 0] = 1.0

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

            # Mean-pool support memory → class prototype [C]
            # Reduces pixel-level noise; single prototype matching is much cleaner
            # than max-pool over 80 noisy pixel-level tokens.
            s_proto = s_rsp.mean(dim=0)                           # [C]

            q_norm = F.normalize(q_rsp.reshape(B, C, N), dim=1)  # [B, C, N]
            s_norm = F.normalize(s_proto, dim=0)                  # [C]

            sim = torch.einsum("bcn,c->bn", q_norm, s_norm)      # [B, N]
            rsp_map = sim.reshape(B, 1, H, W)                     # [B, 1, H, W]

            rsp_min = rsp_map.amin(dim=(2, 3), keepdim=True)
            rsp_max = rsp_map.amax(dim=(2, 3), keepdim=True)
            rsp_map = (rsp_map - rsp_min) / (rsp_max - rsp_min + 1e-5)
        else:
            rsp_map = torch.zeros(B, 1, H, W, device=query_features.device)

        # ═════════════════════════════════════════════════════
        # Merge: rsp_map → geometric prior [1, C, H, W]
        # ═════════════════════════════════════════════════════
        geometric_prior = self.rsp_expand(rsp_map)  # [1, C, H, W]

        return geometric_prior
