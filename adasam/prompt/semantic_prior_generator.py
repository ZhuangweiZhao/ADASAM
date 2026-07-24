"""
语义先验生成器 | Semantic Prior Generator (SPG).
=================================================

AdaSAM 核心创新模块: 从 query 特征 + support memory 生成统一的语义先验。

Core innovation: generates a unified semantic prior from query features +
support memory, replacing the old DPG's N-query multi-proposal interface.

架构说明 | Architecture:
    SPG 内部使用 N 个可学习语义探针 (semantic probes, Mask2Former 式 masked
    cross-attention) 作为实现细节, 但对外只暴露统一的 semantic prior + prior_mask。
    旧 DPG 暴露 fg_queries [N,C]、fg_logits [N]、fg_mask_logits [N,gh,gw] 等
    多 query 接口, 现在全部内部聚合后输出单一先验。

    SPG uses N learnable semantic probes internally (Mask2Former-style masked
    cross-attention) as an implementation detail, but only exposes a unified
    semantic prior + prior_mask externally.

每层顺序 | Per-layer order:
    masked cross-attn (image) → support cross-attn → self-attn → FFN, post-norm.

参考 | Reference:
    - Cheng et al., "Masked-attention Mask Transformer", CVPR 2022.
    - SAM-RSP: Support Representation → Prompt Generator paradigm.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SemanticPriorGeneratorConfig:
    """SPG 配置 | SPG configuration.

    :param num_probes: 内部语义探针数 N (实现细节, 不暴露) | internal semantic probe count N.
    :param embed_dim: 特征维度 (SAM token 维) | feature dim (SAM token dim).
    :param num_layers: 解码层数 L | number of decoder layers L.
    :param num_heads: 注意力头数 | attention heads.
    :param ffn_dim: FFN 隐层维度 | FFN hidden dim.
    :param dropout: dropout 概率 | dropout probability.
    :param use_feedback: 层间 mask feedback (SAM-RSP style).
    """

    num_probes: int = 16
    embed_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.0
    use_feedback: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "SemanticPriorGeneratorConfig":
        known = {f.name for f in fields(cls)}
        # Accept legacy key "num_queries" as alias for "num_probes"
        if "num_queries" in d and "num_probes" not in d:
            d = {**d, "num_probes": d["num_queries"]}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class SPGOutput:
    """SPG 前向输出 | SPG forward output.

    只暴露统一语义先验, 不暴露任何 per-probe 内部状态。
    Only exposes unified semantic prior; no per-probe internal state.

    :param semantic_prior: [1, C, gh, gw] 统一语义先验 | unified semantic prior.
    :param prior_mask: [1, 1, gh, gw] 或 None. prior 投影粗掩码, L_prior 监督目标.
    :param prior_aux: L 个中间 prior snapshot {"prior_mask": [1, gh, gw]}.
        每层的 unified prior mask, 用于 deep supervision。
    """

    semantic_prior: torch.Tensor
    prior_mask: torch.Tensor | None
    prior_aux: list[dict[str, torch.Tensor]]


class _MLP(nn.Module):
    """3 层 MLP (Mask2Former probe_proj 形制)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


class SemanticPriorGenerator(nn.Module):
    """语义先验生成器 | Semantic prior generator.

    内部使用 N 个可学习语义探针通过 masked cross-attention 探索前景区域,
    聚合后对外只暴露统一的 semantic prior + prior_mask。
    Internally uses N learnable semantic probes via masked cross-attention,
    aggregates to expose only unified semantic prior + prior_mask externally.

    职责 | Responsibility:
        query_features + support_memory → semantic_prior + prior_mask

    Dense prompt / sparse token 由 PromptFusion 生产, 不再由 SPG 负责。
    Dense prompt / sparse token are produced by PromptFusion, not by SPG.

    :param cfg: :class:`SemanticPriorGeneratorConfig`.
    """

    def __init__(self, cfg: SemanticPriorGeneratorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c, n, layers = cfg.embed_dim, cfg.num_probes, cfg.num_layers

        # ── 内部可学习语义探针 (实现细节) | internal learnable semantic probes ──
        self.probe_feat = nn.Embedding(n, c)
        self.probe_pos = nn.Embedding(n, c)

        # ── Image cross-attention (masked) ──
        self.cross_attn = nn.ModuleList(
            nn.MultiheadAttention(c, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
            for _ in range(layers)
        )
        self.cross_norm = nn.ModuleList(nn.LayerNorm(c) for _ in range(layers))

        # ── Support cross-attention ──
        self.cross_attn_support = nn.ModuleList(
            nn.MultiheadAttention(c, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
            for _ in range(layers)
        )
        self.support_cross_norm = nn.ModuleList(nn.LayerNorm(c) for _ in range(layers))

        # ── Self-attention (probe ↔ probe interaction) ──
        self.self_attn = nn.ModuleList(
            nn.MultiheadAttention(c, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
            for _ in range(layers)
        )
        self.self_norm = nn.ModuleList(nn.LayerNorm(c) for _ in range(layers))

        # ── FFN ──
        self.ffn = nn.ModuleList(
            nn.Sequential(
                nn.Linear(c, cfg.ffn_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.ffn_dim, c),
            )
            for _ in range(layers)
        )
        self.ffn_norm = nn.ModuleList(nn.LayerNorm(c) for _ in range(layers))

        # ── Inter-layer mask feedback (SAM-RSP style) ──
        # 用 per-probe mask 的统计信息更新 mask_features,
        # 使下一层的探针能看到上一层的发现。
        # Updates mask_features with per-probe mask statistics,
        # allowing probes in the next layer to benefit from previous discoveries.
        if cfg.use_feedback and layers > 1:
            self.feedback_conv = nn.ModuleList(
                nn.Conv2d(c + 2, c, kernel_size=1, bias=False)
                for _ in range(layers - 1)
            )
            for conv in self.feedback_conv:
                weight = torch.zeros(c, c + 2, 1, 1)
                for j in range(c):
                    weight[j, j, 0, 0] = 1.0
                conv.weight.data.copy_(weight)
        else:
            self.feedback_conv = None

        # ── Per-probe prediction heads (内部实现, 不暴露) ──
        self.decoder_norm = nn.LayerNorm(c)
        self.probe_weight = nn.Linear(c, 1)            # per-probe confidence
        self.probe_proj = _MLP(c, c, c, num_layers=3)  # probe → mask space projection

        # ── Prior head: mask_features → unified semantic prior [1, C, gh, gw] ──
        self.prior_head = nn.Sequential(
            nn.GroupNorm(num_groups=min(32, c), num_channels=c),
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 1),
        )

        # ── Prior mask head: semantic_prior → single-channel logits [1, 1, gh, gw] ──
        # L_prior 监督目标 — 直接监督 unified semantic prior 的投影
        self.prior_mask_head = nn.Sequential(
            nn.GroupNorm(num_groups=min(32, c), num_channels=c),
            nn.Conv2d(c, 1, 1),
        )
        nn.init.xavier_uniform_(self.prior_mask_head[-1].weight, gain=1.0)
        nn.init.zeros_(self.prior_mask_head[-1].bias)

    # ── Internal per-probe prediction ──

    def _predict(
        self, probes: torch.Tensor, mask_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-probe prediction: (probes [N,C], mask_features [C,gh,gw]) → (probe_logits [N], masks [N,gh,gw]).

        每个语义探针独立预测其关注的区域及置信度。
        Each semantic probe independently predicts its region of interest and confidence.
        """
        pn = self.decoder_norm(probes)
        probe_logits = self.probe_weight(pn)[:, 0]                        # [N]
        masks = torch.einsum("qc,chw->qhw", self.probe_proj(pn), mask_features)  # [N, gh, gw]
        return probe_logits, masks

    @staticmethod
    def _build_attn_mask(mask_logits: torch.Tensor, num_heads: int) -> torch.Tensor:
        """Mask prediction → cross-attention mask (Mask2Former rule).

        每个探针只关注其预测区域内的像素, 实现 masked attention。
        Each probe only attends to pixels within its predicted region.
        """
        attn_mask = (mask_logits.detach().sigmoid() < 0.5).flatten(1)
        degenerate = attn_mask.sum(dim=-1) == attn_mask.shape[-1]
        attn_mask[degenerate] = False
        return attn_mask.unsqueeze(0).expand(num_heads, -1, -1)

    # ── Semantic prior projection ──

    def _project_semantic_prior(
        self, mask_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """将反馈调制后的特征图投影为统一语义先验 | Project feedback-modulated features to unified prior.

        探针发现已通过 feedback_conv 逐层累积到 mask_features 中,
        此处 prior_head 直接将其投影为 semantic prior + prior_mask。
        Probe discoveries have been accumulated into mask_features via feedback_conv;
        prior_head directly projects them to semantic prior + prior_mask.

        :param mask_features: [C, gh, gw] 最终特征图 (经 L 层 feedback 调制).
        :return: (semantic_prior [1,C,gh,gw], prior_mask [1,1,gh,gw]).
        """
        semantic_prior = self.prior_head(mask_features.unsqueeze(0))  # [1, C, gh, gw]
        prior_mask = self.prior_mask_head(semantic_prior)             # [1, 1, gh, gw]
        return semantic_prior, prior_mask

    # ── Unified prior from per-probe masks ──

    def _unify_probe_masks(
        self, masks: torch.Tensor, probe_logits: torch.Tensor
    ) -> torch.Tensor:
        """将 N 个 per-probe mask 聚合为统一 prior mask | Unify N probe masks into one.

        softmax 加权聚合 — 高置信度探针贡献更多。
        Softmax-weighted aggregation — higher-confidence probes contribute more.

        :param masks: [N, gh, gw] per-probe mask logits.
        :param probe_logits: [N] per-probe confidence.
        :return: unified prior mask [1, gh, gw].
        """
        weights = torch.softmax(probe_logits, dim=0)  # [N]
        unified = (masks.sigmoid() * weights[:, None, None]).sum(dim=0)  # [gh, gw]
        return unified.unsqueeze(0)  # [1, gh, gw]

    # ── Forward ──

    def forward(
        self,
        query_features: torch.Tensor,
        support_memory: torch.Tensor,
        dense_pe: torch.Tensor,
    ) -> SPGOutput:
        """前向传播: query_features + support_memory → semantic_prior + prior_mask.

        :param query_features: [1, C, gh, gw] CAT-adapted query features.
        :param support_memory: [M, C] support memory tokens (from SupportEncoder).
        :param dense_pe: [1, C, gh, gw] SAM positional encoding.
        :return: :class:`SPGOutput` with semantic_prior, prior_mask, prior_aux.
        """
        assert query_features.shape[0] == 1, \
            f"SemanticPriorGenerator only supports batch_size=1, got {query_features.shape[0]}"
        gh, gw = query_features.shape[2], query_features.shape[3]

        mask_features = query_features[0]                          # [C, gh, gw]
        memory = query_features.flatten(2).permute(0, 2, 1)        # [1, gh*gw, C]
        memory_pe = dense_pe.flatten(2).permute(0, 2, 1)           # [1, gh*gw, C]
        probe_pos = self.probe_pos.weight.unsqueeze(0)             # [1, N, C]

        q = self.probe_feat.weight.unsqueeze(0)                    # [1, N, C]

        has_support = support_memory.shape[0] > 0
        if has_support:
            support_key = support_memory.unsqueeze(0)              # [1, M, C]
        else:
            support_key = None

        prior_aux: list[dict[str, torch.Tensor]] = []
        probe_logits, masks = self._predict(q[0], mask_features)

        for i in range(self.cfg.num_layers):
            # Store unified prior snapshot for deep supervision (after aggregation)
            prior_aux.append({
                "prior_mask": self._unify_probe_masks(masks, probe_logits)
            })

            attn_mask = self._build_attn_mask(masks, self.cfg.num_heads)

            # 1. masked cross-attention (probes → image features) [WHERE]
            out, _ = self.cross_attn[i](
                query=q + probe_pos, key=memory + memory_pe, value=memory,
                attn_mask=attn_mask, need_weights=False,
            )
            q = self.cross_norm[i](q + out)

            # 2. support cross-attention (probes → support memory) [WHAT]
            if has_support:
                out_s, _ = self.cross_attn_support[i](
                    query=q + probe_pos, key=support_key, value=support_key,
                    need_weights=False,
                )
                q = self.support_cross_norm[i](q + out_s)

            # 3. self-attention (probe ↔ probe interaction)
            out, _ = self.self_attn[i](
                query=q + probe_pos, key=q + probe_pos, value=q, need_weights=False,
            )
            q = self.self_norm[i](q + out)

            # 4. FFN
            q = self.ffn_norm[i](q + self.ffn[i](q))

            probe_logits, masks = self._predict(q[0], mask_features)

            # Inter-layer mask feedback
            # 每层探针的发现通过 feedback_conv 更新 mask_features,
            # 使 prior_head 能综合所有层的发现生成 semantic_prior。
            if self.feedback_conv is not None and i < self.cfg.num_layers - 1:
                masks_prob = masks.detach().sigmoid()
                fg_conf = masks_prob.max(dim=0)[0]                    # max over probes
                probe_w = probe_logits.detach().sigmoid()
                weighted = (masks_prob * probe_w[:, None, None]).sum(dim=0)  # weighted sum

                extra = torch.stack([fg_conf, weighted], dim=0).unsqueeze(0)
                feedback = self.feedback_conv[i](
                    torch.cat([mask_features.unsqueeze(0), extra], dim=1)
                )[0]
                mask_features = mask_features + feedback

        # ── Project feedback-modulated mask_features → unified semantic prior ──
        # 探针发现已通过 feedback_conv 累积到 mask_features, prior_head 直接投影。
        # Probe discoveries accumulated in mask_features via feedback_conv;
        # prior_head directly projects to unified semantic prior.
        semantic_prior, prior_mask = self._project_semantic_prior(mask_features)

        return SPGOutput(
            semantic_prior=semantic_prior,
            prior_mask=prior_mask,
            prior_aux=prior_aux,
        )
