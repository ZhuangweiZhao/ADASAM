"""
原型驱动的 SAM 掩码解码器 | Prototype-prompted SAM mask decoder.
================================================================

AdaSAM 的核心解码模块。它复用 MobileSAM (=SAM) 的 PromptEncoder + MaskDecoder, 通过
"原型→点提示" 让每个提示点解码出一个实例掩码 (PerSAM/Matcher 范式)。类信息经一个可学习的
PrototypeAdapter 注入到每个提示的 sparse token 中。
The core decoding module. It reuses MobileSAM's PromptEncoder + MaskDecoder, and via
"prototype → point prompts" makes each point decode one instance mask (PerSAM/Matcher). Class
information is injected into every prompt's sparse tokens by a learnable PrototypeAdapter.

契约 | Contract::

    forward(image_embedding[1,256,64,64], prototype[256]) -> (masks[N,H,W] bool, scores[N])

训练 (teacher forcing) 与推理共用 decode() 核心 —— 训练用 GT 实例内采样点, 推理用相似度峰值,
无 if-mode 分支。decode() core is shared by training (GT points) and inference (sim peaks).

可训练面 | Trainable surface:
    - PrototypeAdapter (始终可训练, 末层零初始化 → 初始≈纯 PerSAM)。
    - MaskDecoder (可选, 默认可训练); PromptEncoder 默认冻结。
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from adasam.prototype.matcher import Matcher, similarity_map


class InstanceMasks(NamedTuple):
    """实例分割输出 | Instance segmentation output.

    :param masks: [N, H, W] bool 每实例掩码 | per-instance binary masks.
    :param scores: [N] float 置信度 ∈ [0,1] | per-instance confidence.
    """

    masks: torch.Tensor
    scores: torch.Tensor


class PrototypeAdapter(nn.Module):
    """原型 → 可学习提示 token | Prototype → learnable prompt token(s).

    末层零初始化: 初始输出零 token, 使解码器起始等价于纯点提示 PerSAM; 训练学习类条件贡献。
    Final layer zero-initialized: outputs a zero token at start, so the decoder begins as pure
    point-prompt PerSAM; training learns the class-conditional contribution.

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


class PromptMaskDecoder(nn.Module):
    """原型驱动的 SAM 掩码解码器 | Prototype-prompted SAM mask decoder.

    :param prompt_encoder: MobileSAM PromptEncoder (来自同一 Sam) | from the same Sam.
    :param mask_decoder: MobileSAM MaskDecoder (来自同一 Sam) | from the same Sam.
    :param embed_dim: 提示/原型维度 | prompt/prototype dim.
    :param image_size: 编码器输入边长 (1024) | encoder input side length.
    :param top_k_points / sim_threshold / min_distance: Matcher 配置 | matcher config.
    :param n_proto_tokens: 注入的原型 token 数 | number of injected prototype tokens.
    :param train_mask_decoder / train_prompt_encoder: 各 SAM 部件是否可训练 | trainable flags.
    """

    mask_threshold: float = 0.0

    def __init__(
        self,
        prompt_encoder: nn.Module,
        mask_decoder: nn.Module,
        embed_dim: int = 256,
        image_size: int = 1024,
        top_k_points: int = 10,
        sim_threshold: float = 0.5,
        min_distance: int = 1,
        n_proto_tokens: int = 1,
        train_mask_decoder: bool = True,
        train_prompt_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.proto_adapter = PrototypeAdapter(embed_dim, n_proto_tokens)
        self.matcher = Matcher(top_k_points, sim_threshold, min_distance)

        self.image_size = int(image_size)
        gh, gw = prompt_encoder.image_embedding_size            # (64, 64)
        self.grid_size = (int(gh), int(gw))
        self.stride = self.image_size / gh                       # 1024/64 = 16

        self._set_trainable(self.prompt_encoder, train_prompt_encoder)
        self._set_trainable(self.mask_decoder, train_mask_decoder)

    @staticmethod
    def _set_trainable(module: nn.Module, trainable: bool) -> None:
        for p in module.parameters():
            p.requires_grad_(trainable)

    # ── 核心: 由显式点提示解码 (训练/推理共用) | Core: decode from explicit points ──

    def decode(
        self,
        image_embedding: torch.Tensor,
        prototype: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """由点提示解码低分辨率掩码 logits | Decode low-res mask logits from point prompts.

        :param image_embedding: [1, C, gh, gw] 单图嵌入 | single-image embedding.
        :param prototype: [C] 类原型 | class prototype.
        :param point_coords: [N, 2] 输入帧 (x,y) | input-frame point coords.
        :param point_labels: [N] (1=正) | point labels.
        :return: (low_res_logits[N, 1, 256, 256], iou_pred[N, 1]).
        """
        if image_embedding.ndim != 4 or image_embedding.shape[0] != 1:
            raise ValueError(f"expected image_embedding [1,C,gh,gw], got {tuple(image_embedding.shape)}")

        sparse, dense = self.prompt_encoder(
            points=(point_coords[:, None, :], point_labels[:, None]), boxes=None, masks=None
        )                                                        # [N,2,C], [N,C,gh,gw]

        proto_tokens = self.proto_adapter(prototype)             # [T, C]
        proto_tokens = proto_tokens.unsqueeze(0).expand(sparse.shape[0], -1, -1)  # [N,T,C]
        sparse = torch.cat([proto_tokens, sparse], dim=1)        # [N, T+2, C]

        low_res, iou_pred = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )                                                        # [N,1,256,256], [N,1]
        return low_res, iou_pred

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

    # ── 推理: 由原型相似度峰值生成提示 | Inference: prompts from prototype-sim peaks ──

    @torch.no_grad()
    def forward(
        self,
        image_embedding: torch.Tensor,
        prototype: torch.Tensor,
        input_size: tuple[int, int] = (1024, 1024),
        original_size: tuple[int, int] = (896, 896),
    ) -> InstanceMasks:
        """图像嵌入 + 原型 → 每实例掩码与置信度 | Embedding + prototype → per-instance masks & scores.

        :return: InstanceMasks(masks[N,H,W] bool, scores[N] float ∈ [0,1]).
        """
        sim = similarity_map(image_embedding[0], prototype)      # [gh, gw]
        pts = self.matcher.select(sim, stride=self.stride)

        low_res, iou_pred = self.decode(image_embedding, prototype, pts.coords, pts.labels)
        logits = self.upscale_logits(low_res, input_size, original_size)   # [N, H, W]

        masks = logits > self.mask_threshold                     # bool
        scores = iou_pred[:, 0].clamp(0.0, 1.0) * pts.sims.clamp(0.0, 1.0)
        return InstanceMasks(masks=masks, scores=scores)

    # ── 坐标帧变换辅助 (供 trainer 把 GT tile 坐标映射到输入帧) ──
    # Coordinate-frame helper (for trainer: map GT tile coords → input frame).

    @staticmethod
    def scale_points(
        coords_xy: torch.Tensor,
        from_size: tuple[int, int],
        to_size: tuple[int, int],
    ) -> torch.Tensor:
        """把 (x,y) 从 from_size 帧线性映射到 to_size 帧 | linearly map (x,y) between frames.

        :param coords_xy: [N, 2] (x, y).
        :param from_size: 源尺寸 (H, W) | source (H, W).
        :param to_size: 目标尺寸 (H, W) | target (H, W).
        """
        sx = to_size[1] / from_size[1]
        sy = to_size[0] / from_size[0]
        scale = torch.tensor([sx, sy], device=coords_xy.device, dtype=coords_xy.dtype)
        return coords_xy * scale
