"""
粗先验模块 | Coarse Prior Module (SAM-RSP inspired).
======================================================

受 SAM-RSP 启发的两阶段粗先验模块，在 DPG 之前丰富查询特征:

  Stage 1 — RSP (Rough Segmentation Prompt):
    support memory 与 query features 的余弦相似度 → 空间先验图,
    指示"目标大概在哪里"。
    Cosine similarity between support memory and query features
    → spatial prior map suggesting "where the target roughly is".

  Stage 2 — Pixel-level Query Prototype:
    查询特征自相关 + RSP 门控 → 动态查询原型,
    捕获"查询图像中哪些像素彼此相似"。
    Query feature self-correlation gated by RSP → dynamic prototype
    capturing intra-image feature similarity patterns.

输出: [query_features, pixel_prototype, rsp_map] 拼接融合 → 丰富后的特征,
送入 DPG 做精细化实例发现。
Output: [query_features, pixel_prototype, rsp_map] merged → enriched features
for fine-grained DPG instance discovery.

参考 | Reference:
    - SAM-RSP: "Representation Prompting for SAM-based Few-shot Segmentation"
      (thirdparty/SAM-RSP, model/SAM_RSP.py).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoarsePriorModule(nn.Module):
    """SAM-RSP 风格的粗先验 + 像素级查询原型生成器.

    Generates a coarse spatial prior from support-query similarity (RSP),
    then computes a pixel-level query prototype via self-correlation gating.
    The enriched features feed into DPG for fine-grained instance discovery.

    :param embed_dim: 特征维度 (SAM token dim, 256) | feature dimension.
    """

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        C = embed_dim

        # ---- Stage 1: RSP projection layers ----
        self.query_rsp_proj = nn.Conv2d(C, C, kernel_size=1, bias=False)
        self.support_rsp_proj = nn.Linear(C, C, bias=False)

        # ---- Stage 2: Merge [query_features, pixel_prototype, rsp_map] → enriched ----
        # 输入 | input: C + C + 1 = 2C + 1 channels
        self.merge = nn.Sequential(
            nn.Conv2d(C + C + 1, C, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        query_features: torch.Tensor,
        support_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """生成粗先验并丰富查询特征 | Generate coarse prior and enrich query features.

        :param query_features: [1, C, gh, gw] 查询图像特征 | query image features.
        :param support_memory: [M, C] support memory tokens; 可为空 [0, C].
        :return: ``(enriched_features [1, C, gh, gw], rsp_map [1, 1, gh, gw])``.
        """
        B, C, H, W = query_features.shape
        N = H * W
        has_support = support_memory.shape[0] > 0

        # ═══════════════════════════════════════════════════════════
        # Stage 1: RSP — 从 support-query 相似度生成粗空间先验
        #           Coarse spatial prior from support-query similarity
        # ═══════════════════════════════════════════════════════════
        if has_support:
            q_rsp = self.query_rsp_proj(query_features)          # [B, C, H, W]
            s_rsp = self.support_rsp_proj(support_memory)        # [M, C]

            q_norm = F.normalize(q_rsp.reshape(B, C, N), dim=1)  # [B, C, N]
            s_norm = F.normalize(s_rsp, dim=1)                    # [M, C]

            # 每个空间位置对 memory tokens 的最大余弦相似度
            # Max cosine similarity over memory tokens per spatial position
            sim = torch.einsum("bcn,mc->bmn", q_norm, s_norm)    # [B, M, N]
            sim = sim.max(dim=1)[0]                                # [B, N]
            rsp_map = sim.reshape(B, 1, H, W)                     # [B, 1, H, W]

            # Min-max 归一化到 [0, 1] (per-image, 不考虑 batch)
            rsp_min = rsp_map.amin(dim=(2, 3), keepdim=True)
            rsp_max = rsp_map.amax(dim=(2, 3), keepdim=True)
            rsp_map = (rsp_map - rsp_min) / (rsp_max - rsp_min + 1e-5)
        else:
            rsp_map = torch.zeros(B, 1, H, W, device=query_features.device)

        # ═══════════════════════════════════════════════════════════
        # Stage 2: 像素级查询原型 (自相关 + RSP 门控)
        #           Pixel-level Query Prototype via self-correlation
        # ═══════════════════════════════════════════════════════════
        q_flat = query_features.reshape(B, C, N)                   # [1, C, N]

        # 缩放点积自相关 | Scaled dot-product self-correlation
        corr = torch.bmm(q_flat.permute(0, 2, 1), q_flat)          # [1, N, N]
        corr = corr / math.sqrt(C)

        # 逐行归一化 | Row-wise normalize
        corr_min = corr.min(dim=2, keepdim=True)[0]
        corr_max = corr.max(dim=2, keepdim=True)[0]
        corr = (corr - corr_min) / (corr_max - corr_min + 1e-7)

        # RSP 门控: 只有 RSP 有前景置信度的位置才能贡献
        # RSP gate: only foreground-confident positions contribute
        gate = rsp_map.reshape(B, 1, N).clamp(min=0.1)            # [B, 1, N]
        corr = corr * gate
        corr = F.threshold(corr, threshold=0.1, value=-1e7)
        corr = F.softmax(corr, dim=-1)                              # [1, N, N]

        # 加权求和 → 像素级原型 | Weighted sum → pixel-level prototype
        pixel_proto = torch.bmm(corr, q_flat.permute(0, 2, 1))     # [1, N, C]
        pixel_proto = pixel_proto.permute(0, 2, 1).reshape(
            B, C, H, W
        )                                                           # [1, C, H, W]

        # ═══════════════════════════════════════════════════════════
        # Merge: [query, prototype, rsp] → enriched features
        # ═══════════════════════════════════════════════════════════
        enriched = self.merge(
            torch.cat([query_features, pixel_proto, rsp_map], dim=1)
        )

        return enriched, rsp_map
