"""
密集提示生成器 | Dense Prompt Generator (DPG).
================================================

核心创新模块: 以可学习 Instance Query + 掩码交叉注意力 (Mask2Former 式) 从查询特征中
直接生成 N 个实例查询嵌入, 取代旧的 "相似度峰 → 点提示" 定位路径。
The core innovation: learnable instance queries + masked cross-attention
(Mask2Former-style) generate N instance query embeddings directly from query
features, replacing the legacy "similarity peaks → point prompts" localization.

设计 | Design:
    - 原型仅作语义条件 (零初始化投影后加到每个 query 上), 不负责定位。
      The prototype is a semantic condition only (zero-init projection added to
      every query); it does NOT localize.
    - 每层顺序 (Mask2Former 签名): masked cross-attn → self-attn → FFN, post-norm。
      Per-layer order (Mask2Former signature): masked cross-attn → self-attn → FFN.
    - 注意力掩码来自上一次掩码预测 (sigmoid < 0.5 阻断, detach, 全阻断行守卫)。
      Attention masks come from the previous mask prediction (sigmoid < 0.5
      blocked, detached, degenerate all-blocked-row guard).
    - L+1 次预测 (初始 + 每层后); 前 L 次作为深监督 aux, 最后一次为主输出。
      L+1 predictions (initial + after each layer); first L are deep-supervision
      aux, the last is the main output.
    - dense_pe 由调用方传入 (prompt_encoder.get_dense_pe()), 本模块零依赖 SAM。
      dense_pe is supplied by the caller; this module has zero SAM dependency.

参考 | Reference:
    Cheng et al., "Masked-attention Mask Transformer for Universal Image
    Segmentation", CVPR 2022 (thirdparty/Mask2Former, MIT).
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
    """

    instance_queries: torch.Tensor
    objectness_logits: torch.Tensor
    mask_logits: torch.Tensor
    aux: list[dict[str, torch.Tensor]]


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

        # 原型语义条件投影 (零初始化 → 训练起点与原型无关, 仓库惯例)
        # prototype semantic-condition projection (zero-init → prototype-agnostic
        # at init; repo convention, same as CATAdapter.up / PrototypeAdapter.fc2)
        self.proto_proj = nn.Linear(c, c)
        nn.init.zeros_(self.proto_proj.weight)
        nn.init.zeros_(self.proto_proj.bias)

        # L 层: masked cross-attn → self-attn → FFN (post-norm)
        self.cross_attn = nn.ModuleList(
            nn.MultiheadAttention(c, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
            for _ in range(layers)
        )
        self.cross_norm = nn.ModuleList(nn.LayerNorm(c) for _ in range(layers))
        self.self_attn = nn.ModuleList(
            nn.MultiheadAttention(c, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
            for _ in range(layers)
        )
        self.self_norm = nn.ModuleList(nn.LayerNorm(c) for _ in range(layers))
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

        # 共享预测头 (L+1 次预测复用同一组) | shared prediction heads (reused L+1 times)
        self.decoder_norm = nn.LayerNorm(c)
        self.obj_head = nn.Linear(c, 1)
        self.mask_embed = _MLP(c, c, c, num_layers=3)

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
        return attn_mask.unsqueeze(0).repeat(num_heads, 1, 1)

    # ── 前向 | Forward ──

    def forward(
        self,
        query_features: torch.Tensor,
        prototype: torch.Tensor,
        dense_pe: torch.Tensor,
    ) -> DPGOutput:
        """前向传播 | Forward pass.

        :param query_features: [1, C, gh, gw] CAT 适配后的查询特征 | CAT-adapted features.
        :param prototype: [C] L2 归一化原型 (可为全零) | L2-normalized prototype (may be zeros).
        :param dense_pe: [1, C, gh, gw] SAM 位置编码 | SAM dense positional encoding.
        :return: :class:`DPGOutput`.
        """
        mask_features = query_features[0]                          # [C, gh, gw]
        memory = query_features.flatten(2).permute(0, 2, 1)        # [1, gh*gw, C]
        memory_pe = dense_pe.flatten(2).permute(0, 2, 1)           # [1, gh*gw, C]
        query_pos = self.query_pos.weight.unsqueeze(0)             # [1, N, C]

        # 初始查询 = 可学习内容 + 原型语义条件 | initial queries = content + condition
        q = (self.query_feat.weight + self.proto_proj(prototype).unsqueeze(0)).unsqueeze(0)

        aux: list[dict[str, torch.Tensor]] = []
        objectness, masks = self._predict(q[0], mask_features)     # 预测 0 | prediction 0

        for i in range(self.cfg.num_layers):
            aux.append({"mask_logits": masks, "objectness_logits": objectness})
            attn_mask = self._build_attn_mask(masks, self.cfg.num_heads)

            # 1. masked cross-attention (query → image memory)
            out, _ = self.cross_attn[i](
                query=q + query_pos, key=memory + memory_pe, value=memory,
                attn_mask=attn_mask, need_weights=False,
            )
            q = self.cross_norm[i](q + out)
            # 2. self-attention (query ↔ query)
            out, _ = self.self_attn[i](
                query=q + query_pos, key=q + query_pos, value=q, need_weights=False,
            )
            q = self.self_norm[i](q + out)
            # 3. FFN
            q = self.ffn_norm[i](q + self.ffn[i](q))

            objectness, masks = self._predict(q[0], mask_features)  # 预测 i+1

        return DPGOutput(
            instance_queries=self.decoder_norm(q[0]),
            objectness_logits=objectness,
            mask_logits=masks,
            aux=aux,
        )
