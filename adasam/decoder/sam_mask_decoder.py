"""
查询驱动的 SAM 掩码解码器 | Query-prompted SAM mask decoder.
==============================================================

将 DensePromptGenerator 产出的 N 个实例查询作为 sparse prompt token 喂入
MobileSAM 原始 MaskDecoder (原样复用, 每个 query 即一个独立 prompt set)。
Feeds the N instance queries from the DensePromptGenerator into the original
MobileSAM MaskDecoder as sparse prompt tokens (reused unchanged; each query is
one independent prompt set).

设计 | Design:
    - 无点提示、无框提示 — sparse = [N, 1, C] 纯查询 token (可选前置原型 token)。
      No point/box prompts — sparse = [N, 1, C] pure query tokens
      (optionally prefixed with a prototype token).
    - dense = no_mask_embed (SAM "无掩码提示" 的预训练嵌入), 保持解码器输入分布。
      dense = no_mask_embed (SAM's pretrained "no mask prompt" embedding).
    - PromptEncoder 永远冻结; MaskDecoder 默认可训练。
      PromptEncoder always frozen; MaskDecoder trainable by default.
    - decode_chunk_size > 0 时分块顺序解码, 限制 N 较大时的峰值显存。
      Chunked sequential decoding bounds peak memory for large N.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class QueryMaskDecoderConfig:
    """QueryMaskDecoder 配置 | configuration.

    :param embed_dim: 提示 token 维度 (256) | prompt token dim.
    :param image_size: 编码器输入边长 (1024) | encoder input side length.
    :param use_proto_token: 是否前置零初始化原型 token | prepend zero-init prototype token.
    :param train_mask_decoder: MaskDecoder 是否可训练 | MaskDecoder trainable.
    :param decode_chunk_size: 0 = 一次解码全部 N; >0 = 分块顺序解码 (限显存)。
        0 = decode all N at once; >0 = sequential chunks (bounds peak memory).
    """

    embed_dim: int = 256
    image_size: int = 1024
    use_proto_token: bool = True
    train_mask_decoder: bool = True
    decode_chunk_size: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "QueryMaskDecoderConfig":
        """从 yaml 字典构建, 忽略未知键 | build from a yaml dict, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class PrototypeAdapter(nn.Module):
    """原型 → 可学习提示 token | Prototype → learnable prompt token(s).

    末层零初始化: 初始输出零 token, 使解码器起始行为不受原型影响; 训练学习类条件贡献。
    Final layer zero-initialized: outputs a zero token at start, so the decoder
    begins prototype-agnostic; training learns the class-conditional contribution.

    :param embed_dim: 提示 token 维度 (256) | prompt token dim.
    :param n_tokens: 注入的 token 数 | number of injected tokens.
    :param hidden_dim: MLP 隐藏维 | MLP hidden dim.
    """

    def __init__(self, embed_dim: int = 256, n_tokens: int = 1, hidden_dim: int = 256) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_tokens = n_tokens
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, n_tokens * embed_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, prototype: torch.Tensor) -> torch.Tensor:
        """:param prototype: [embed_dim]; :return: [n_tokens, embed_dim]."""
        x = self.fc2(self.act(self.fc1(prototype)))
        return x.view(self.n_tokens, self.embed_dim)


class QueryMaskDecoder(nn.Module):
    """实例查询 → SAM 掩码 | Instance queries → SAM masks.

    :param prompt_encoder: MobileSAM PromptEncoder (仅用 get_dense_pe / no_mask_embed,
        永远冻结) | only get_dense_pe / no_mask_embed used; always frozen.
    :param mask_decoder: MobileSAM MaskDecoder (原样复用) | reused unchanged.
    :param cfg: :class:`QueryMaskDecoderConfig`.
    """

    mask_threshold: float = 0.0

    def __init__(
        self,
        prompt_encoder: nn.Module,
        mask_decoder: nn.Module,
        cfg: QueryMaskDecoderConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.proto_adapter = (
            PrototypeAdapter(cfg.embed_dim) if cfg.use_proto_token else None
        )

        self.image_size = int(cfg.image_size)
        gh, gw = prompt_encoder.image_embedding_size             # (64, 64)
        self.grid_size = (int(gh), int(gw))

        self._set_trainable(self.prompt_encoder, False)          # 永远冻结 | always frozen
        self._set_trainable(self.mask_decoder, cfg.train_mask_decoder)

    @staticmethod
    def _set_trainable(module: nn.Module, trainable: bool) -> None:
        for p in module.parameters():
            p.requires_grad_(trainable)

    # ── 解码 | Decode ──

    def forward(
        self,
        image_embedding: torch.Tensor,
        instance_queries: torch.Tensor,
        prototype: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """N 个实例查询 → N 个低分辨率掩码 | N instance queries → N low-res masks.

        :param image_embedding: [1, C, gh, gw] 单图嵌入 | single-image embedding.
        :param instance_queries: [N, C] 实例查询 | instance queries.
        :param prototype: [C] 类原型 (仅 use_proto_token 时使用) | class prototype.
        :return: (low_res_logits [N, 1, 256, 256], iou_pred [N, 1]).
        """
        if image_embedding.ndim != 4 or image_embedding.shape[0] != 1:
            raise ValueError(
                f"expected image_embedding [1,C,gh,gw], got {tuple(image_embedding.shape)}"
            )
        n = instance_queries.shape[0]

        sparse = instance_queries.unsqueeze(1)                   # [N, 1, C]
        if self.proto_adapter is not None:
            if prototype is None:
                raise ValueError("use_proto_token=True requires a prototype")
            proto_tok = self.proto_adapter(prototype)            # [T, C]
            sparse = torch.cat(
                [proto_tok.unsqueeze(0).expand(n, -1, -1), sparse], dim=1
            )                                                    # [N, T+1, C]

        chunk = self.cfg.decode_chunk_size
        if chunk <= 0 or n <= chunk:
            return self._decode(image_embedding, sparse)

        lows, ious = [], []
        for start in range(0, n, chunk):
            low, iou = self._decode(image_embedding, sparse[start : start + chunk])
            lows.append(low)
            ious.append(iou)
        return torch.cat(lows, dim=0), torch.cat(ious, dim=0)

    def _decode(
        self, image_embedding: torch.Tensor, sparse: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """单块解码 | Decode one chunk of prompt sets."""
        n = sparse.shape[0]
        gh, gw = self.grid_size
        # SAM "无掩码提示" 的预训练稠密嵌入 | SAM's pretrained "no mask prompt" embedding
        dense = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            n, -1, gh, gw
        )
        low_res, iou_pred = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )                                                        # [N,1,256,256], [N,1]
        return low_res, iou_pred

    # ── 上采样 | Upscale ──

    def upscale_logits(
        self,
        low_res_logits: torch.Tensor,
        input_size: tuple[int, int],
        original_size: tuple[int, int],
    ) -> torch.Tensor:
        """低分辨率 logits → 原图尺寸 logits | Low-res logits → original-size logits.

        与 SAM.postprocess_masks 语义一致: 上采样到 image_size → 去 padding → 缩放到原尺寸。
        Same as SAM.postprocess_masks: upscale to image_size → remove padding → resize to original.

        :return: [N, H, W] logits (原尺寸) | logits at original size.
        """
        x = F.interpolate(
            low_res_logits, (self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        )
        x = x[..., : input_size[0], : input_size[1]]             # 去 padding | remove padding
        x = F.interpolate(x, original_size, mode="bilinear", align_corners=False)
        return x[:, 0]                                           # [N, H, W]
