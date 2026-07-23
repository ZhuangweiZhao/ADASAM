"""
密集提示生成器 | Dense Prompt Generator (DPG).
================================================

核心创新模块: 以可学习 Instance Query + 掩码交叉注意力 (Mask2Former 式) 从查询特征中
直接生成 N 个实例查询嵌入, 取代旧的 "相似度峰 → 点提示" 定位路径。
The core innovation: learnable instance queries + masked cross-attention
(Mask2Former-style) generate N instance query embeddings directly from query
features, replacing the legacy "similarity peaks → point prompts" localization.

v2 设计 | v2 Design (SAM-RSP inspired):
    - Instance Queries 同时 attend to query image features (Where) 和 support
      memory tokens (What), 实现解耦的定位+条件识别。
      Instance queries attend to BOTH query image features (Where) and support
      memory tokens (What) for decoupled localization + class conditioning.
    - Support cross-attention 插在每层 masked image cross-attn 之后。
      Support cross-attention inserted after each layer's masked image cross-attn.
    - Support-conditioned Dense Prompt: 从 support memory 生成 dense prompt,
      替代 SAM 的通用 no_mask_embed。
      Support-conditioned dense prompt generated from support memory,
      replacing SAM's generic no_mask_embed.

每层顺序 | Per-layer order (v2):
    masked cross-attn (image) → support cross-attn → self-attn → FFN, post-norm.

参考 | Reference:
    - Cheng et al., "Masked-attention Mask Transformer for Universal Image
      Segmentation", CVPR 2022 (thirdparty/Mask2Former, MIT).
    - SAM-RSP: Support Representation → Prompt Generator paradigm.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn as nn


@dataclass(frozen=True)
class DensePromptGeneratorConfig:
    """DPG 配置 | DPG configuration.

    :param num_queries: 实例查询数 N | number of instance queries N.
    :param embed_dim: 特征维度 (SAM token 维) | feature dim (SAM token dim).
    :param num_layers: 解码层数 L | number of decoder layers L.
    :param num_heads: 注意力头数 | attention heads.
    :param ffn_dim: FFN 隐层维度 | FFN hidden dim.
    :param dropout: dropout 概率 | dropout probability.
    """

    num_queries: int = 64
    embed_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.0
    use_feedback: bool = True  # SAM-RSP 式层间特征回传 | inter-layer mask feedback

    @classmethod
    def from_dict(cls, d: dict) -> "DensePromptGeneratorConfig":
        """从 yaml 字典构建, 忽略未知键 | build from a yaml dict, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class DPGOutput:
    """DPG 前向输出 | DPG forward output.

    :param instance_queries: [N, C] 最终实例查询 (已过 decoder_norm) | final queries.
    :param objectness_logits: [N] 最终 objectness | final objectness logits.
    :param mask_logits: [N, gh, gw] 最终 64² 内部掩码 (aux 监督) | final internal masks.
    :param aux: L 个中间预测 {"mask_logits": [N,gh,gw], "objectness_logits": [N]}。
        L intermediate predictions for deep supervision.
    :param dense_prompt: [1, C, gh, gw] 或 None. support-conditioned dense prompt
        (替换 SAM 的 no_mask_embed); None 时回退到 no_mask_embed。
    :param prompt_mask: [1, 1, gh, gw] 或 None. V3 新增: dense prompt 投影粗掩码,
        用于辅助 BCE 监督 (prompt auxiliary mask loss).
    """

    instance_queries: torch.Tensor
    objectness_logits: torch.Tensor
    mask_logits: torch.Tensor
    aux: list[dict[str, torch.Tensor]]
    dense_prompt: torch.Tensor | None = None
    prompt_mask: torch.Tensor | None = None


class _MLP(nn.Module):
    """3 层 MLP (Mask2Former mask_embed 形制) | 3-layer MLP (Mask2Former mask_embed)."""

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


class DensePromptGenerator(nn.Module):
    """密集提示生成器 | Dense prompt generator.

    (query_features, prototype, dense_pe) → N 个实例查询 + objectness + 64² aux 掩码。
    (query_features, prototype, dense_pe) → N instance queries + objectness +
    64² aux masks.
    """

    def __init__(self, cfg: DensePromptGeneratorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c, n, layers = cfg.embed_dim, cfg.num_queries, cfg.num_layers

        # 可学习查询 (内容 + 位置) | learnable queries (content + positional)
        self.query_feat = nn.Embedding(n, c)
        self.query_pos = nn.Embedding(n, c)

        # ---- Image cross-attention (masked) ----
        self.cross_attn = nn.ModuleList(
            nn.MultiheadAttention(c, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
            for _ in range(layers)
        )
        self.cross_norm = nn.ModuleList(nn.LayerNorm(c) for _ in range(layers))

        # ---- Support cross-attention (NEW: query reads support "what" info) ----
        self.cross_attn_support = nn.ModuleList(
            nn.MultiheadAttention(c, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
            for _ in range(layers)
        )
        self.support_cross_norm = nn.ModuleList(nn.LayerNorm(c) for _ in range(layers))

        # ---- Self-attention ----
        self.self_attn = nn.ModuleList(
            nn.MultiheadAttention(c, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
            for _ in range(layers)
        )
        self.self_norm = nn.ModuleList(nn.LayerNorm(c) for _ in range(layers))

        # ---- FFN ----
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

        # ---- Inter-layer mask feedback (SAM-RSP style) ----
        # 每层中间预测 → 空间置信度图 → 拼回 mask_features 作为下一层条件
        # Each layer's intermediate mask → spatial confidence → fed back to
        # mask_features for next-layer conditioning.
        if cfg.use_feedback and layers > 1:
            self.feedback_conv = nn.ModuleList(
                nn.Conv2d(c + 2, c, kernel_size=1, bias=False)
                for _ in range(layers - 1)
            )
            # identity-init: first C channels pass through, extra 2 channels start at zero.
            # Residual connection (mask_features = mask_features + conv(...)) ensures
            # feedback starts as identity and gradually incorporates mask information.
            for conv in self.feedback_conv:
                weight = torch.zeros(c, c + 2, 1, 1)
                for j in range(c):
                    weight[j, j, 0, 0] = 1.0
                conv.weight.data.copy_(weight)
        else:
            self.feedback_conv = None

        # ---- Shared prediction heads (reused L+1 times) ----
        self.decoder_norm = nn.LayerNorm(c)
        self.obj_head = nn.Linear(c, 1)
        self.mask_embed = _MLP(c, c, c, num_layers=3)

        # ---- Spatial Dense Prompt Generator (V3) ----
        # 从 support 特征图生成空间 dense prompt [C, H, W], 取代旧的全局 [C, 1, 1].
        # support_features [K, C, H, W] × support_masks [K, H, W] → masked avg →
        # conv spatial projector → spatial prompt [1, C, H, W].
        # Generate spatial dense prompt from support feature maps, replacing the old
        # global [C, 1, 1] vector. support spatial features contain mask-conditioned
        # spatial structure that tells SAM decoder WHERE (not just WHETHER) to segment.
        self.spatial_prompt_proj = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
        )
        nn.init.xavier_uniform_(self.spatial_prompt_proj[-1].weight, gain=1.0)
        nn.init.zeros_(self.spatial_prompt_proj[-1].bias)

        # Learnable scale for spatial prompt. Plan A (no no_mask_embed) removes
        # the need for conservative init — scale=1.0 gives the spatial signal
        # enough magnitude to drive prompt_mask_head learning (previously stuck
        # at sigmoid(0)=0.5 because input was ~0.0003).
        # 可学习缩放因子: Plan A 无需保守初始化, scale=1.0 给空间信号足够的幅度
        # 驱动 prompt_mask_head 学习 (之前因为输入≈0.0003 卡在 sigmoid(0)=0.5)。
        self.spatial_prompt_scale = nn.Parameter(torch.tensor(1.0))

        # ---- Prompt auxiliary mask head (V3 BCE supervision) ----
        # 1×1 Conv 将 dense prompt [1,C,H,W] 投影为粗掩码 logits [1,1,H,W],
        # 用于辅助 BCE+Dice 监督 — 迫使 dense prompt 学习类别判别的空间激活。
        # 1×1 Conv projects dense prompt to a single-channel coarse mask logits,
        # supervised by BCE+Dice against GT union — forces the dense prompt
        # to learn class-discriminative spatial activation patterns.
        self.prompt_mask_head = nn.Conv2d(c, 1, 1)
        nn.init.xavier_uniform_(self.prompt_mask_head.weight, gain=1.0)
        nn.init.zeros_(self.prompt_mask_head.bias)

        # Legacy: global dense prompt (kept for backward compat when no spatial input)
        self.dense_pool_attn = nn.Linear(c, 1)
        self.dense_prompt_gen = nn.Sequential(
            nn.Linear(c, c),
            nn.ReLU(inplace=True),
            nn.Linear(c, c),
        )
        nn.init.xavier_uniform_(self.dense_prompt_gen[-1].weight, gain=1.0)
        nn.init.zeros_(self.dense_prompt_gen[-1].bias)

    # ── 预测头 | Prediction heads ──

    def _predict(
        self, queries: torch.Tensor, mask_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(queries [N,C], mask_features [C,gh,gw]) → (objectness [N], masks [N,gh,gw])."""
        qn = self.decoder_norm(queries)
        objectness = self.obj_head(qn)[:, 0]
        masks = torch.einsum("qc,chw->qhw", self.mask_embed(qn), mask_features)
        return objectness, masks

    @staticmethod
    def _build_attn_mask(mask_logits: torch.Tensor, num_heads: int) -> torch.Tensor:
        """掩码预测 → 交叉注意力掩码 | mask prediction → cross-attention mask.

        Mask2Former 规则: sigmoid < 0.5 处阻断 (True); 全阻断行退化守卫解除阻断;
        detach — 注意力掩码不回传梯度。
        Mask2Former rule: blocked (True) where sigmoid < 0.5; degenerate
        all-blocked rows are fully unblocked; detached (no gradient through it).

        :param mask_logits: [N, gh, gw] 掩码 logits | mask logits.
        :return: [num_heads, N, gh*gw] bool, True = 阻断 | True = blocked.
        """
        attn_mask = (mask_logits.detach().sigmoid() < 0.5).flatten(1)  # [N, gh*gw]
        degenerate = attn_mask.sum(dim=-1) == attn_mask.shape[-1]
        attn_mask[degenerate] = False
        # MHA with batch_first=True expects [B*num_heads, L, S]; B=1 → [num_heads, N, gh*gw]
        return attn_mask.unsqueeze(0).expand(num_heads, -1, -1)

    # ── 前向 | Forward ──

    def forward(
        self,
        query_features: torch.Tensor,
        support_memory: torch.Tensor,
        dense_pe: torch.Tensor,
        support_features: torch.Tensor | None = None,
        support_masks_grid: torch.Tensor | None = None,
    ) -> DPGOutput:
        """前向传播 | Forward pass.

        :param query_features: [1, C, gh, gw] CAT 适配后的查询特征 | CAT-adapted features.
        :param support_memory: [M, C] support memory tokens (from SupportEncoder).
            可为空 [0, C] — 此时跳过 support cross-attention, DPG 退化为无条件模式。
        :param dense_pe: [1, C, gh, gw] SAM 位置编码 | SAM dense positional encoding.
        :param support_features: [K, C, gh, gw] (V3 新增) support 空间特征图,
            用于生成空间 dense prompt。None 时退化为全局 prompt。
            (V3 new) support spatial feature maps for spatial dense prompt.
        :param support_masks_grid: [K, gh, gw] (V3 新增) support FG masks (已 resize).
        :return: :class:`DPGOutput`.
        """
        assert query_features.shape[0] == 1, \
            f"DensePromptGenerator only supports batch_size=1, got {query_features.shape[0]}"
        gh, gw = query_features.shape[2], query_features.shape[3]
        mask_features = query_features[0]                          # [C, gh, gw]
        memory = query_features.flatten(2).permute(0, 2, 1)        # [1, gh*gw, C]
        memory_pe = dense_pe.flatten(2).permute(0, 2, 1)           # [1, gh*gw, C]
        query_pos = self.query_pos.weight.unsqueeze(0)             # [1, N, C]

        # 初始查询 = 可学习内容 (无 prototype 条件) | initial = content only (no prototype)
        q = self.query_feat.weight.unsqueeze(0)                 # [1, N, C]

        # 将 support_memory 扩展 batch 维度给 cross-attention
        has_support = support_memory.shape[0] > 0
        if has_support:
            support_key = support_memory.unsqueeze(0)              # [1, M, C]
        else:
            support_key = None

        aux: list[dict[str, torch.Tensor]] = []
        objectness, masks = self._predict(q[0], mask_features)     # 预测 0

        for i in range(self.cfg.num_layers):
            aux.append({"mask_logits": masks, "objectness_logits": objectness})
            attn_mask = self._build_attn_mask(masks, self.cfg.num_heads)

            # 1. masked cross-attention (query → image features)  [WHERE]
            out, _ = self.cross_attn[i](
                query=q + query_pos, key=memory + memory_pe, value=memory,
                attn_mask=attn_mask, need_weights=False,
            )
            q = self.cross_norm[i](q + out)

            # 2. support cross-attention (query → support memory)  [WHAT]  ← NEW
            if has_support:
                out_s, _ = self.cross_attn_support[i](
                    query=q + query_pos, key=support_key, value=support_key,
                    need_weights=False,
                )
                q = self.support_cross_norm[i](q + out_s)

            # 3. self-attention (query ↔ query)
            out, _ = self.self_attn[i](
                query=q + query_pos, key=q + query_pos, value=q, need_weights=False,
            )
            q = self.self_norm[i](q + out)

            # 4. FFN
            q = self.ffn_norm[i](q + self.ffn[i](q))

            objectness, masks = self._predict(q[0], mask_features)  # 预测 i+1

            # ---- Inter-layer mask feedback (SAM-RSP style) ----
            # 将当前层的掩码预测作为额外空间条件拼入 mask_features,
            # 下一层可基于"已发现的位置"进一步精修。
            # Feed current mask prediction back as extra spatial conditioning
            # into mask_features, so the next layer can refine further.
            if self.feedback_conv is not None and i < self.cfg.num_layers - 1:
                # 前景置信度 (max over queries) + 加权掩码 (objectness-weighted)
                # Foreground confidence (max over queries) + weighted mask
                masks_prob = masks.detach().sigmoid()                      # [N, gh, gw]
                fg_conf = masks_prob.max(dim=0)[0]                         # [gh, gw]
                obj_w = objectness.detach().sigmoid()                       # [N]
                weighted = (masks_prob * obj_w[:, None, None]).sum(dim=0)   # [gh, gw]

                extra = torch.stack([fg_conf, weighted], dim=0).unsqueeze(0)  # [1, 2, gh, gw]
                # Residual: identity-init conv ensures unchanged features at start;
                # feedback channels gradually add mask-conditioned spatial modulation.
                feedback = self.feedback_conv[i](
                    torch.cat([mask_features.unsqueeze(0), extra], dim=1)
                )[0]                                                       # [C, gh, gw]
                mask_features = mask_features + feedback                   # residual! identity-like at t=0

        # ---- Dense prompt generation ----
        # V3 spatial path (preferred): support features × masks → spatial prompt [1,C,gh,gw]
        # V2 legacy path (fallback): support memory → global prompt [1,C,1,1]
        prompt_mask = None  # auxiliary mask from dense_prompt projection
        if has_support and support_features is not None and support_masks_grid is not None:
            # ── V3 Spatial Dense Prompt ──
            # Mask × features → per-support masked features, then mean-pool
            # across K support images. Each (c,h,w) position gets the average
            # target feature at that spatial position. No sqrt division —
            # preserves natural feature magnitude so conv layers operate at
            # their designed scale.
            masked = support_features * support_masks_grid.unsqueeze(1)    # [K, C, gh, gw]
            support_spatial = masked.mean(dim=0, keepdim=True)              # [1, C, gh, gw]
            dense_prompt = self.spatial_prompt_proj(support_spatial)       # [1, C, gh, gw]
            dense_prompt = self.spatial_prompt_scale * dense_prompt        # learnable scale
            prompt_mask = self.prompt_mask_head(dense_prompt)              # [1, 1, gh, gw]
        elif has_support:
            # ── V2 Legacy: Global Dense Prompt (fallback) ──
            attn_scores = self.dense_pool_attn(support_memory)    # [M, 1]
            attn_weights = torch.softmax(attn_scores, dim=0)      # [M, 1]
            support_summary = (support_memory * attn_weights).sum(dim=0)  # [C]
            dense_mod = self.dense_prompt_gen(support_summary)     # [C]
            dense_prompt = dense_mod.view(1, -1, 1, 1)           # [1, C, 1, 1]
        else:
            dense_prompt = None

        return DPGOutput(
            instance_queries=self.decoder_norm(q[0]),
            objectness_logits=objectness,
            mask_logits=masks,
            aux=aux,
            dense_prompt=dense_prompt,
            prompt_mask=prompt_mask,
        )
