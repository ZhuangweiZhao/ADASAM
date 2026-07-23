"""
Support Representation Encoder | 支持表征编码器.
=================================================

将 K 张 support 图像的特征与其前景掩码编码为一组 support memory tokens,
替代原有的 Mean Prototype (256-d 单一向量), 保留空间结构信息。
Encode K support image features + foreground masks into a set of support
memory tokens, replacing the legacy Mean Prototype (single 256-d vector)
and preserving spatial structure.

两阶段设计 | Two-stage design:
    Stage 1 (MVP, n_encoder_layers=0):
        1. Masked Token Extraction: 在每张 support 的 FG 区域内采样 N_s 个位置
        2. Concatenation: 直接拼接 K × N_s 个 tokens 作为 support memory
           输出: [K × N_s, C]

    Stage 2 (增强, n_encoder_layers>0):
        1. Masked Token Extraction (同 Stage 1)
        2. Support Self-Attention: L_s 层 TransformerEncoder, 所有 support tokens
           通过 self-attention 相互交互
        3. Memory Bank: M 个可学习 memory tokens 通过 cross-attention 从编码后的
           support tokens 中压缩读取信息
           输出: [M, C] 固定大小

设计参考 | References:
    - SAM-RSP: "Representation Prompting for SAM-based Few-shot Segmentation"
    - Perceiver: "Perceiver: General Perception with Iterative Attention"
    - DETR / Mask2Former: learned query + cross-attention decoder patterns
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SupportEncoderConfig:
    """Support Encoder 配置 | configuration.

    :param embed_dim: 特征维度 (256, SAM token dim) | feature dimension.
    :param n_support_tokens: 每张 support 采样的 token 数 N_s | tokens per support.
    :param n_memory_tokens: support memory 的最终 token 数 M | final memory token count.
        - 0 或不启用 Stage 2 时: M = K × N_s (直接拼接)
        - Stage 2 启用时: 压缩到固定 M 个 memory tokens
    :param n_encoder_layers: Support self-attention 层数 L_s | encoder layers.
        0 = Stage 1 MVP (无 self-attention); >0 = Stage 2 (TransformerEncoder).
    :param n_heads: 注意力头数 | attention heads.
    :param ffn_dim: FFN 隐藏维 | FFN hidden dim.
    :param dropout: dropout 概率 | dropout probability.
    :param max_support_images: support 索引 embedding 的词典大小 | vocab size for index embedding.
        必须 ≥ 最大 K-shot 值。默认 50 覆盖 1-shot 到 50-shot 所有配置。
    """

    embed_dim: int = 256
    n_support_tokens: int = 16
    n_memory_tokens: int = 64
    n_encoder_layers: int = 0          # 0 = MVP; 2 = Stage 2
    n_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.0
    max_support_images: int = 50       # support idx embedding vocab size

    @classmethod
    def from_dict(cls, d: dict) -> "SupportEncoderConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def is_stage2(self) -> bool:
        """是否启用 Stage 2 增强 | whether Stage 2 enhancements are enabled."""
        return self.n_encoder_layers > 0


# ---------------------------------------------------------------------------
# Memory Bank — Perceiver-style cross-attention compression
# ---------------------------------------------------------------------------

class _MemoryBank(nn.Module):
    """Memory Bank: 可学习 memory tokens 通过 cross-attention 压缩 support tokens.

    Learnable memory tokens compress support tokens via cross-attention
    (Perceiver-style: memory tokens act as queries, support tokens as KV).

    :param embed_dim: token 维度 | token dim.
    :param n_memory_tokens: M 个 memory tokens | number of memory tokens.
    :param n_heads: attention heads.
    :param ffn_dim: FFN hidden dim.
    :param dropout: dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_memory_tokens: int = 64,
        n_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        C = embed_dim
        self.n_memory_tokens = n_memory_tokens

        # 可学习的 memory tokens | learnable memory tokens
        self.memory_tokens = nn.Parameter(torch.zeros(1, n_memory_tokens, C))
        nn.init.normal_(self.memory_tokens, std=0.02)

        # Cross-attention: memory (Q) → support (K, V)
        self.cross_attn = nn.MultiheadAttention(
            C, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(C)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(C, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, C),
        )
        self.ffn_norm = nn.LayerNorm(C)

    def forward(self, support_tokens: torch.Tensor) -> torch.Tensor:
        """压缩 K×N_s support tokens → M memory tokens.

        Compress K×N_s support tokens → M memory tokens.

        :param support_tokens: [K×N_s, C] 编码后的 support tokens.
        :return: [M, C] compressed memory tokens.
        """
        M = self.n_memory_tokens
        # 扩展 batch 维 | expand batch dim
        q = self.memory_tokens                                   # [1, M, C]
        kv = support_tokens.unsqueeze(0)                         # [1, K×N_s, C]

        # Cross-attention: memory tokens query support tokens
        out, _ = self.cross_attn(query=q, key=kv, value=kv, need_weights=False)
        q = self.cross_norm(q + out)                             # [1, M, C]

        # FFN
        q = self.ffn_norm(q + self.ffn(q))                       # [1, M, C]

        return q[0]                                              # [M, C]


# ---------------------------------------------------------------------------
# SupportEncoder
# ---------------------------------------------------------------------------

class SupportEncoder(nn.Module):
    """Support Representation Encoder.

    将 K 张 support 特征 + 掩码编码为 support memory tokens。
    Encodes K support features + masks into support memory tokens.

    Stage 1 (MVP, n_encoder_layers=0):
        Extract → Concat → [K×N_s, C]

    Stage 2 (Enhanced, n_encoder_layers>0):
        Extract → Self-Attention (L_s layers) → MemoryBank → [M, C]

    :param cfg: :class:`SupportEncoderConfig`.
    """

    def __init__(self, cfg: SupportEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        C = cfg.embed_dim

        # ---- positional encoding for sampled positions ----
        self.pos_proj = nn.Sequential(
            nn.Linear(2, C),
            nn.ReLU(inplace=True),
            nn.Linear(C, C),
        )
        if isinstance(self.pos_proj[-1], nn.Linear):
            nn.init.zeros_(self.pos_proj[-1].weight)
            nn.init.zeros_(self.pos_proj[-1].bias)

        # ---- learnable [MASK] token for padding ----
        self.mask_token = nn.Parameter(torch.zeros(1, C))
        nn.init.normal_(self.mask_token, std=0.02)

        # ---- support index embedding (区分不同 support image) ----
        self.support_idx_embed = nn.Embedding(cfg.max_support_images, C)

        # ---- Stage 2: Support Self-Attention (TransformerEncoder) ----
        if cfg.is_stage2:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=C,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.ffn_dim,
                dropout=cfg.dropout,
                activation="relu",
                batch_first=True,
                norm_first=True,              # pre-norm for stability
            )
            self.support_self_attn = nn.TransformerEncoder(
                encoder_layer, num_layers=cfg.n_encoder_layers
            )

            # ---- Stage 2: Memory Bank (compression) ----
            if cfg.n_memory_tokens > 0:
                self.memory_bank = _MemoryBank(
                    embed_dim=C,
                    n_memory_tokens=cfg.n_memory_tokens,
                    n_heads=cfg.n_heads,
                    ffn_dim=cfg.ffn_dim,
                    dropout=cfg.dropout,
                )
            else:
                self.memory_bank = None
        else:
            self.support_self_attn = None
            self.memory_bank = None

    # ------------------------------------------------------------------
    # Masked Token Extraction (shared across Stage 1 and Stage 2)
    # ------------------------------------------------------------------

    def _extract_tokens(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """从单张 support 的 FG 区域提取 N_s 个 tokens。

        Extract N_s tokens from the FG region of one support image.

        :param features: [C, gh, gw] 特征图 | feature map.
        :param mask: [gh, gw] 二值前景掩码 | binary foreground mask.
        :return: (tokens [N_s, C], valid_mask [N_s] bool).
        """
        C, gh, gw = features.shape
        device = features.device
        N_s = self.cfg.n_support_tokens

        fg_coords = mask.nonzero(as_tuple=False)                  # [n_fg, 2] (y, x)
        n_fg = fg_coords.shape[0]

        tokens = []
        valid = []

        if n_fg >= N_s:
            indices = torch.linspace(0, n_fg - 1, N_s, device=device).long()
            indices = torch.clamp(indices, 0, n_fg - 1)
            for idx in indices:
                y, x = fg_coords[idx]
                feat = features[:, y, x]                          # [C]
                self._add_token(tokens, valid, C, device, feat, y, x, gh, gw, is_valid=True)
        elif n_fg > 0:
            for i in range(n_fg):
                y, x = fg_coords[i]
                feat = features[:, y, x]
                self._add_token(tokens, valid, C, device, feat, y, x, gh, gw, is_valid=True)
            n_pad = N_s - n_fg
            for _ in range(n_pad):
                self._add_token(tokens, valid, C, device, None, 0, 0, gh, gw, is_valid=False)
        else:
            for _ in range(N_s):
                self._add_token(tokens, valid, C, device, None, 0, 0, gh, gw, is_valid=False)

        return torch.stack(tokens, dim=0), torch.tensor(valid, device=device)

    def _add_token(
        self,
        tokens: list,
        valid: list,
        C: int,
        device: torch.device,
        feat: torch.Tensor | None,
        y: int,
        x: int,
        gh: int,
        gw: int,
        is_valid: bool,
    ) -> None:
        """构建单个 token: feature + positional encoding (或 [MASK])."""
        if is_valid and feat is not None:
            coord = torch.tensor(
                [(x + 0.5) / gw, (y + 0.5) / gh],
                device=device, dtype=torch.float32,
            )
            pe = self.pos_proj(coord)                             # [C]
            token = feat + pe
        else:
            token = self.mask_token[0]                            # [C]
        tokens.append(token)
        valid.append(is_valid)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        support_features: torch.Tensor,
        support_masks: torch.Tensor,
    ) -> torch.Tensor:
        """编码 Support → Support Memory Tokens。

        Encode support features → support memory tokens.

        :param support_features: [K, C, gh, gw] K 张 support 特征图.
        :param support_masks: [K, gh, gw] K 张二值前景掩码 (已 resize 到特征图尺寸).
        :return: support_memory [M, C].
            Stage 1: M = K × N_s
            Stage 2: M = cfg.n_memory_tokens (if > 0), else K × N_s
        """
        if support_features.ndim != 4:
            raise ValueError(
                f"expected support_features [K,C,gh,gw], got {tuple(support_features.shape)}"
            )
        K, C, gh, gw = support_features.shape

        if support_masks.ndim != 3:
            raise ValueError(
                f"expected support_masks [K,gh,gw], got {tuple(support_masks.shape)}"
            )

        # Step 1: Extract tokens from each support
        all_tokens: list[torch.Tensor] = []
        for k in range(K):
            feats = support_features[k]                           # [C, gh, gw]
            mask = support_masks[k]                               # [gh, gw]
            tokens, _valid = self._extract_tokens(feats, mask)    # [N_s, C]

            idx_emb = self.support_idx_embed(
                torch.tensor(k, device=feats.device)
            )                                                     # [C]
            tokens = tokens + idx_emb.unsqueeze(0)                # [N_s, C]
            all_tokens.append(tokens)

        support_tokens = torch.cat(all_tokens, dim=0)             # [K×N_s, C]

        # Step 2: Stage 2 enhancements (if enabled)
        if self.support_self_attn is not None:
            # Support self-attention: all tokens interact
            # [K×N_s, C] → [1, K×N_s, C] (batch_first)
            support_tokens = self.support_self_attn(
                support_tokens.unsqueeze(0)
            )[0]                                                  # [K×N_s, C]

        # Step 3: Memory Bank compression (if enabled)
        if self.memory_bank is not None:
            support_memory = self.memory_bank(support_tokens)     # [M, C]
        else:
            support_memory = support_tokens                       # [K×N_s, C]

        return support_memory
