"""
Correlation Builder | 相关构建器.
==================================

从 Dense Support Features [K, 256, 64, 64] 和 Query Feature [1, 256, 64, 64] 构建
Similarity Tensor [K, 64, 64]。

Builds a Similarity Tensor [K, 64, 64] from Dense Support Features [K, 256, 64, 64]
and Query Feature [1, 256, 64, 64].

关键设计决策 | Key design decision:
    - 输出 K 个独立的 similarity map，**不融合**。融合延后到 Prompt Generator 内部，
      让网络学习每个 support 在每个 region 的权重。
      Outputs K independent similarity maps with NO fusion. Fusion is deferred to the
      Prompt Generator, which learns per-support-per-region weights.

算法 | Algorithm (Choice A — simple cosine + prototype gate):
    对每张 support k:
        sim_k[h,w] = cosine_similarity(support_k[:,h,w], query[:,h,w])
        gate_k = sigmoid(cosine_similarity(support_pooled_k, prototype))
        sim_tensor[k] = sim_k * gate_k

复杂度 | Complexity: O(K × 64 × 64) — negligible.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from adasam.utils.transforms import resize_mask


def similarity_tensor(
    support_features: torch.Tensor,
    prototype: torch.Tensor,
    query_feature: torch.Tensor,
    support_masks: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """构建 Similarity Tensor [K, H, W] | Build Similarity Tensor [K, H, W].

    每张 support 用自己的池化特征 (子原型) 与 query 每个位置做余弦相似度。
    Each support uses its pooled feature as a "sub-prototype" and computes
    cosine similarity against every query spatial location.

    当提供 support_masks 时使用 FG-masked 池化 (与 PrototypeBuilder 一致),
    避免背景特征污染子原型导致热力图反转。
    When support_masks are provided, FG-masked pooling is used (consistent with
    PrototypeBuilder), preventing background contamination that inverts the heatmap.

    :param support_features: [K, C, H, W] dense support embeddings.
    :param prototype: [C] global class prototype (L2-normalized).
    :param query_feature: [1, C, H, W] query image embedding.
    :param support_masks: optional K FG masks at any resolution (resized to grid).
        None → fall back to global mean pooling (backward compatible, but may invert).
    :return: sim_tensor [K, H, W] — one similarity map per support.
    """
    if support_features.ndim != 4:
        raise ValueError(
            f"expected support_features [K, C, H, W], got {tuple(support_features.shape)}"
        )
    if query_feature.ndim != 4 or query_feature.shape[0] != 1:
        raise ValueError(
            f"expected query_feature [1, C, H, W], got {tuple(query_feature.shape)}"
        )

    K, C, H, W = support_features.shape
    device = support_features.device

    # ── Per-support sub-prototype ──
    if support_masks is not None:
        if len(support_masks) != K:
            raise ValueError(
                f"expected {K} support_masks, got {len(support_masks)}"
            )
        # FG-masked pooling (same semantics as PrototypeBuilder)
        pooled_list = []
        for k in range(K):
            m = resize_mask(support_masks[k], (H, W)).to(device)      # [H, W]
            denom = m.sum().clamp(min=1.0)
            pooled = (support_features[k] * m.unsqueeze(0)).sum(dim=(1, 2)) / denom  # [C]
            pooled_list.append(pooled)
        support_pooled = torch.stack(pooled_list, dim=0)               # [K, C]
    else:
        # Global mean pooling (backward compatible, may include BG noise)
        support_pooled = support_features.mean(dim=(2, 3))             # [K, C]

    support_pooled_n = F.normalize(support_pooled, dim=1)              # [K, C]

    # ── Normalize query per spatial location ──
    qf_flat = query_feature[0].reshape(C, -1)                          # [C, H*W]
    qf_n = F.normalize(qf_flat, dim=0)                                 # [C, H*W]

    # ── Cosine similarity: each support's sub-prototype vs each query location ──
    # sim[k, loc] = dot(support_pooled_n[k], qf_n[:, loc])
    sim = torch.einsum("kc,ci->ki", support_pooled_n, qf_n)            # [K, H*W]
    sim = sim.reshape(K, H, W)                                          # [K, H, W]

    # ── Prototype gate: how relevant is support k to the global class prototype? ──
    proto_n = F.normalize(prototype, dim=0)                             # [C]
    gates = torch.sigmoid(
        torch.einsum("kc,c->k", support_pooled_n, proto_n)
    )                                                                    # [K]

    sim_tensor = sim * gates.view(K, 1, 1)                             # [K, H, W]
    return sim_tensor


class CorrelationBuilder:
    """相关构建器 (类形式, 与 Matcher/PrototypeBuilder 风格一致) |
    Correlation builder (class form, consistent with Matcher/PrototypeBuilder style).

    无状态 — 仅包装 similarity_tensor() 以匹配模块约定。
    Stateless — wraps similarity_tensor() to match the module convention.
    """

    def build(
        self,
        support_features: torch.Tensor,
        prototype: torch.Tensor,
        query_feature: torch.Tensor,
        support_masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """构建 Similarity Tensor | Build Similarity Tensor.

        :param support_features: [K, C, H, W].
        :param prototype: [C].
        :param query_feature: [1, C, H, W].
        :param support_masks: optional K FG masks for masked pooling.
        :return: sim_tensor [K, H, W].
        """
        return similarity_tensor(support_features, prototype, query_feature, support_masks)
