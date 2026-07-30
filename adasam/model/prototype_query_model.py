"""Prototype-conditioned query decoder for class-conditioned semantic segmentation."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PrototypeQueryConfig:
    embed_dim: int = 256
    num_queries: int = 16
    num_layers: int = 3
    num_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.0
    fpn_dim: int = 256

    @classmethod
    def from_dict(cls, values: dict) -> "PrototypeQueryConfig":
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})


@dataclass
class PrototypeQueryOutput:
    query_logits: torch.Tensor  # [B, Q], relevance of each semantic query
    query_mask_logits: torch.Tensor  # [B, Q, H, W], semantic component masks
    semantic_logits: torch.Tensor  # [B, H, W], union of all query components
    conditioned_queries: torch.Tensor  # [B, Q, C]
    prototype: torch.Tensor  # [B, C]
    auxiliary: list[dict[str, torch.Tensor]]


class _DecoderLayer(nn.Module):
    def __init__(self, cfg: PrototypeQueryConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_heads
        self.cross_attn = nn.MultiheadAttention(
            cfg.embed_dim, cfg.num_heads, cfg.dropout, batch_first=True
        )
        self.self_attn = nn.MultiheadAttention(
            cfg.embed_dim, cfg.num_heads, cfg.dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_dim, cfg.embed_dim),
        )
        self.norm_cross = nn.LayerNorm(cfg.embed_dim)
        self.norm_self = nn.LayerNorm(cfg.embed_dim)
        self.norm_ffn = nn.LayerNorm(cfg.embed_dim)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        cross, _ = self.cross_attn(
            queries, memory, memory, attn_mask=attention_mask, need_weights=False
        )
        queries = self.norm_cross(queries + cross)
        self_out, _ = self.self_attn(queries, queries, queries, need_weights=False)
        queries = self.norm_self(queries + self_out)
        return self.norm_ffn(queries + self.ffn(queries))


class PrototypeConditionedSemanticQueryDecoder(nn.Module):
    """Query-based semantic decoder conditioned by an explicit support prototype.

    Queries represent semantic components or regions, not object instances. Their
    relevance-weighted masks are combined with a differentiable probabilistic union
    and supervised by one class-level binary semantic mask.
    """

    def __init__(self, cfg: PrototypeQueryConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c, q = cfg.embed_dim, cfg.num_queries
        self.query_embed = nn.Embedding(q, c)
        self.prototype_norm = nn.LayerNorm(c)
        self.prototype_film = nn.Linear(c, 2 * c)
        self.prototype_query_attn = nn.MultiheadAttention(c, cfg.num_heads, cfg.dropout, batch_first=True)
        self.query_norm = nn.LayerNorm(c)
        self.pixel_proj = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.GroupNorm(8, c),
            nn.GELU(),
        )
        self.fpn_lateral = nn.ModuleDict(
            {
                "stage0": nn.Conv2d(64, c, 1),
                "stage1": nn.Conv2d(128, c, 1),
                "stage2": nn.Conv2d(160, c, 1),
                "stage3": nn.Conv2d(256, c, 1),
            }
        )
        self.fpn_output = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.GroupNorm(8, c),
            nn.GELU(),
        )
        self.layers = nn.ModuleList(_DecoderLayer(cfg) for _ in range(cfg.num_layers))
        self.mask_embed = nn.Sequential(nn.Linear(c, c), nn.GELU(), nn.Linear(c, c))
        self.relevance_head = nn.Linear(c, 1)
        nn.init.zeros_(self.prototype_film.weight)
        nn.init.zeros_(self.prototype_film.bias)

    def build_prototype(
        self, support_features: torch.Tensor, support_masks: torch.Tensor
    ) -> torch.Tensor:
        if support_features.ndim != 4 or support_masks.ndim != 3:
            raise ValueError("expected support features [K,C,H,W] and masks [K,H,W]")
        weights = support_masks[:, None].to(support_features.dtype)
        prototype = (support_features * weights).sum(dim=(0, 2, 3))
        prototype = prototype / weights.sum().clamp_min(1.0)
        return self.prototype_norm(prototype)

    @staticmethod
    def aggregate_semantic(
        query_logits: torch.Tensor, query_mask_logits: torch.Tensor
    ) -> torch.Tensor:
        weights = query_logits.softmax(dim=1)[..., None, None]
        probability = (weights * query_mask_logits.sigmoid()).sum(dim=1)
        probability = probability.clamp(1e-6, 1.0 - 1e-6)
        return torch.logit(probability)

    def _predict(
        self, queries: torch.Tensor, pixels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_vectors = self.mask_embed(queries)
        query_masks = torch.einsum("bqc,bchw->bqhw", mask_vectors, pixels)
        query_logits = self.relevance_head(queries).squeeze(-1)
        semantic_logits = self.aggregate_semantic(query_logits, query_masks)
        return query_logits, query_masks, semantic_logits

    def _attention_mask(self, masks: torch.Tensor) -> torch.Tensor:
        blocked = masks.detach().sigmoid().flatten(2) < 0.5
        blocked = blocked & ~blocked.all(dim=-1, keepdim=True)
        return blocked.repeat_interleave(self.cfg.num_heads, dim=0)

    def forward(
        self,
        query_features: torch.Tensor | dict[str, torch.Tensor],
        support_features: torch.Tensor,
        support_masks: torch.Tensor,
    ) -> PrototypeQueryOutput:
        batch_size = query_features.shape[0] if isinstance(query_features, torch.Tensor) else query_features["stage3"].shape[0]
        if batch_size != 1:
            raise ValueError("the episodic decoder currently expects one query image")
        prototype = self.build_prototype(support_features, support_masks).unsqueeze(0)
        gamma, beta = self.prototype_film(prototype).chunk(2, dim=-1)
        base = self.query_embed.weight.unsqueeze(0)
        queries = self.query_norm(base * (1.0 + gamma[:, None]) + beta[:, None])
        if isinstance(query_features, dict):
            target_size = query_features["stage1"].shape[-2:]
            pyramid = self.fpn_lateral["stage3"](query_features["stage3"])
            for name in ("stage2", "stage1"):
                lateral = self.fpn_lateral[name](query_features[name])
                pyramid = torch.nn.functional.interpolate(
                    pyramid, size=lateral.shape[-2:], mode="bilinear", align_corners=False
                ) + lateral
            stage0 = torch.nn.functional.adaptive_avg_pool2d(
                self.fpn_lateral["stage0"](query_features["stage0"]), target_size
            )
            pixels = self.fpn_output(pyramid + stage0)
        else:
            pixels = self.pixel_proj(query_features)
        proto_tokens = prototype[:, None, :]
        proto_delta, _ = self.prototype_query_attn(queries, proto_tokens, proto_tokens, need_weights=False)
        queries = queries + proto_delta
        cosine_bias = torch.nn.functional.cosine_similarity(
            pixels, prototype[:, :, None, None], dim=1
        )
        memory = pixels.flatten(2).transpose(1, 2)
        auxiliary = []
        attention_mask = None
        for layer_index, layer in enumerate(self.layers):
            if attention_mask is None:
                bias = 2.0 * cosine_bias.flatten(1)[:, None, :].expand(-1, queries.shape[1], -1)
                bias = bias.repeat_interleave(self.cfg.num_heads, dim=0)
            else:
                blocked = attention_mask.float() * -1.0e4
                bias = blocked + 2.0 * cosine_bias.flatten(1)[:, None, :].expand(-1, queries.shape[1], -1).repeat_interleave(self.cfg.num_heads, dim=0)
            queries = layer(queries, memory, bias)
            query_logits, query_masks, semantic_logits = self._predict(queries, pixels)
            if layer_index + 1 < len(self.layers):
                auxiliary.append(
                    {
                        "query_logits": query_logits,
                        "query_mask_logits": query_masks,
                        "semantic_logits": semantic_logits,
                    }
                )
                attention_mask = self._attention_mask(query_masks)
        return PrototypeQueryOutput(
            query_logits=query_logits,
            query_mask_logits=query_masks,
            semantic_logits=semantic_logits,
            conditioned_queries=queries,
            prototype=prototype,
            auxiliary=auxiliary,
        )

    @staticmethod
    def semantic_probability(output: PrototypeQueryOutput) -> torch.Tensor:
        return output.semantic_logits.sigmoid()
