"""
SAM 语义掩码解码器 | SAM semantic mask decoder.
=================================================

将 PromptFusion 产出的单 token sparse prompt + dense prompt 喂入
MobileSAM 原始 MaskDecoder, 完成边界细化。

Feeds the single-token sparse prompt + dense prompt from PromptFusion
into MobileSAM's original MaskDecoder for boundary refinement.

设计 | Design:
    - 无点提示、无框提示 — sparse = [1, 1, C] 单 token。
      No point/box prompts — sparse = [1, 1, C] single token.
    - dense: support-conditioned dense prompt (从 PromptFusion 传入).
    - PromptEncoder 永远冻结; MaskDecoder 默认可训练。
      PromptEncoder always frozen; MaskDecoder trainable by default.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SemanticMaskDecoderConfig:
    """SemanticMaskDecoder 配置 | configuration.

    :param embed_dim: 提示 token 维度 (256) | prompt token dim.
    :param image_size: 编码器输入边长 (1024) | encoder input side length.
    :param train_mask_decoder: MaskDecoder 是否可训练 | MaskDecoder trainable.
    """

    embed_dim: int = 256
    image_size: int = 1024
    train_mask_decoder: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "SemanticMaskDecoderConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class SemanticMaskDecoder(nn.Module):
    """SAM 语义掩码解码器 | SAM semantic mask decoder.

    接收单 token sparse prompt + dense prompt, 输出细化后的掩码。
    Receives single-token sparse prompt + dense prompt, outputs refined mask.

    :param prompt_encoder: MobileSAM PromptEncoder (永远冻结).
    :param mask_decoder: MobileSAM MaskDecoder (原样复用).
    :param cfg: :class:`SemanticMaskDecoderConfig`.
    """

    mask_threshold: float = 0.0

    def __init__(
        self,
        prompt_encoder: nn.Module,
        mask_decoder: nn.Module,
        cfg: SemanticMaskDecoderConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        self.image_size = int(cfg.image_size)
        gh, gw = prompt_encoder.image_embedding_size
        self.grid_size = (int(gh), int(gw))

        self._set_trainable(self.prompt_encoder, False)
        self._set_trainable(self.mask_decoder, cfg.train_mask_decoder)

    @staticmethod
    def _set_trainable(module: nn.Module, trainable: bool) -> None:
        for p in module.parameters():
            p.requires_grad_(trainable)

    # ── Decode ──

    def forward(
        self,
        image_embedding: torch.Tensor,
        sparse_token: torch.Tensor,
        dense_prompt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """单 token → 低分辨率掩码 | Single token → low-res mask.

        :param image_embedding: [1, C, gh, gw] single-image embedding.
        :param sparse_token: [1, C] single conditioning token.
        :param dense_prompt: [1, C, gh, gw] or None.
            support-conditioned dense prompt; None → fallback to no_mask_embed.
        :return: (low_res_logits [1, 1, 256, 256], iou_pred [1, 1]).
        """
        if image_embedding.ndim != 4 or image_embedding.shape[0] != 1:
            raise ValueError(
                f"expected image_embedding [1,C,gh,gw], got {tuple(image_embedding.shape)}"
            )

        # sparse = single token [1, 1, C]
        sparse = sparse_token.unsqueeze(0)  # [1, 1, C]

        return self._decode(image_embedding, sparse, dense_prompt)

    def _decode(
        self,
        image_embedding: torch.Tensor,
        sparse: torch.Tensor,
        dense_prompt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode one prompt set."""
        n = sparse.shape[0]
        gh, gw = self.grid_size

        # dense prompt: support-conditioned or fallback to no_mask_embed
        if dense_prompt is not None:
            dense = dense_prompt.expand(n, -1, gh, gw)
        else:
            dense = self.prompt_encoder.no_mask_embed.weight.reshape(
                1, -1, 1, 1
            ).expand(n, -1, gh, gw)

        low_res, iou_pred = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )  # [1, 1, 256, 256], [1, 1]
        return low_res, iou_pred

    # ── Upscale ──

    def upscale_logits(
        self,
        low_res_logits: torch.Tensor,
        input_size: tuple[int, int],
        original_size: tuple[int, int],
    ) -> torch.Tensor:
        """低分辨率 logits → 原图尺寸 logits | Low-res logits → original-size logits.

        :return: [N, H, W] logits (原尺寸).
        """
        x = F.interpolate(
            low_res_logits, (self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        )
        x = x[..., : input_size[0], : input_size[1]]
        x = F.interpolate(x, original_size, mode="bilinear", align_corners=False)
        return x[:, 0]  # [N, H, W]
