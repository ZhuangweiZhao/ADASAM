"""
查询驱动的 SAM 掩码解码器 | Query-prompted SAM mask decoder.
==============================================================

将 DensePromptGenerator 产出的 N 个实例查询作为 sparse prompt token 喂入
MobileSAM 原始 MaskDecoder (原样复用, 每个 query 即一个独立 prompt set)。
Feeds the N instance queries from the DensePromptGenerator into the original
MobileSAM MaskDecoder as sparse prompt tokens (reused unchanged; each query is
one independent prompt set).

设计 | Design:
    - 无点提示、无框提示 — sparse = [N, 1, C] 纯查询 token。
      No point/box prompts — sparse = [N, 1, C] pure query tokens.
    - dense: 优先使用 support-conditioned dense prompt (从 DPG 传入),
      否则回退到 SAM 的 no_mask_embed。
      dense: support-conditioned override from DPG when available;
      falls back to SAM's pretrained no_mask_embed.
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
    :param train_mask_decoder: MaskDecoder 是否可训练 | MaskDecoder trainable.
    :param decode_chunk_size: 0 = 一次解码全部 N; >0 = 分块顺序解码 (限显存)。
        0 = decode all N at once; >0 = sequential chunks (bounds peak memory).
    """

    embed_dim: int = 256
    image_size: int = 1024
    train_mask_decoder: bool = True
    decode_chunk_size: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "QueryMaskDecoderConfig":
        """从 yaml 字典构建, 忽略未知键 | build from a yaml dict, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


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
        dense_prompt_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """N 个实例查询 → N 个低分辨率掩码 | N instance queries → N low-res masks.

        :param image_embedding: [1, C, gh, gw] 单图嵌入 | single-image embedding.
        :param instance_queries: [N, C] 实例查询 | instance queries.
        :param dense_prompt_override: [1, C, gh, gw] 或 None.
            support-conditioned dense prompt; None 时回退到 SAM 的 no_mask_embed.
        :return: (low_res_logits [N, 1, 256, 256], iou_pred [N, 1]).
        """
        if image_embedding.ndim != 4 or image_embedding.shape[0] != 1:
            raise ValueError(
                f"expected image_embedding [1,C,gh,gw], got {tuple(image_embedding.shape)}"
            )
        n = instance_queries.shape[0]

        # 稀疏 = 实例查询 (无 proto_adapter 前缀) | sparse = instance queries only
        sparse = instance_queries.unsqueeze(1)                   # [N, 1, C]

        chunk = self.cfg.decode_chunk_size
        if chunk <= 0 or n <= chunk:
            return self._decode(image_embedding, sparse, dense_prompt_override)

        lows, ious = [], []
        for start in range(0, n, chunk):
            low, iou = self._decode(
                image_embedding, sparse[start : start + chunk], dense_prompt_override
            )
            lows.append(low)
            ious.append(iou)
        return torch.cat(lows, dim=0), torch.cat(ious, dim=0)

    def _decode(
        self,
        image_embedding: torch.Tensor,
        sparse: torch.Tensor,
        dense_prompt_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """单块解码 | Decode one chunk of prompt sets."""
        n = sparse.shape[0]
        gh, gw = self.grid_size

        # dense prompt: 优先使用 support-conditioned override, 否则回退到 no_mask_embed
        # prefer support-conditioned override; fall back to SAM's no_mask_embed
        if dense_prompt_override is not None:
            # override shape: [1, C, gh, gw], expand to [N, C, gh, gw]
            dense = dense_prompt_override.expand(n, -1, gh, gw)   # [N, C, gh, gw]
        else:
            dense = self.prompt_encoder.no_mask_embed.weight.reshape(
                1, -1, 1, 1
            ).expand(n, -1, gh, gw)                               # [N, C, gh, gw]

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
