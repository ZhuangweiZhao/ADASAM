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


def similarity_tensor(
    support_features: torch.Tensor,
    prototype: torch.Tensor,
    query_feature: torch.Tensor,
) -> torch.Tensor:
    """构建 Similarity Tensor [K, H, W] | Build Similarity Tensor [K, H, W].

    :param support_features: [K, C, H, W] dense support embeddings.
    :param prototype: [C] global class prototype (L2-normalized).
    :param query_feature: [1, C, H, W] query image embedding.
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

    # ── Normalize prototype once ──
    proto_n = F.normalize(prototype, dim=0)  # [C]

    # ── Per-location cosine similarity (batched over K) ──
    # Normalize each spatial position to unit vector in channel dim
    sf_flat = support_features.reshape(K, C, -1)              # [K, C, H*W]
    sf_n = F.normalize(sf_flat, dim=1)                         # [K, C, H*W]

    qf_flat = query_feature[0].reshape(C, -1)                  # [C, H*W]
    qf_n = F.normalize(qf_flat, dim=0)                         # [C, H*W]

    # Cosine similarity per location: dot product over channel dim
    sim = torch.einsum("kci,ci->ki", sf_n, qf_n)               # [K, H*W]
    sim = sim.reshape(K, H, W)                                  # [K, H, W]

    # ── Prototype gate per support ──
    support_pooled = support_features.mean(dim=(2, 3))         # [K, C]
    support_pooled_n = F.normalize(support_pooled, dim=1)      # [K, C]
    gates = torch.sigmoid(
        torch.einsum("kc,c->k", support_pooled_n, proto_n)
    )                                                            # [K]

    sim_tensor = sim * gates.view(K, 1, 1)                     # [K, H, W]
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
    ) -> torch.Tensor:
        """构建 Similarity Tensor | Build Similarity Tensor.

        :param support_features: [K, C, H, W].
        :param prototype: [C].
        :param query_feature: [1, C, H, W].
        :return: sim_tensor [K, H, W].
        """
        return similarity_tensor(support_features, prototype, query_feature)
