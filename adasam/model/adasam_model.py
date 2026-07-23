"""
AdaSAM 组合模型 | AdaSAM composite model.
===========================================

DensePromptGenerator + SupportEncoder + QueryMaskDecoder 的组合模块 — trainer 与
evaluator 共用同一构建路径, checkpoint "model" 键即本模块的 state_dict。
Composite of DensePromptGenerator + SupportEncoder + QueryMaskDecoder — the single
construction path shared by trainer and evaluator; the checkpoint "model" key is
this module's state_dict.

前向 | Forward:
    forward_train(): 训练用, 返回 DPG 输出 + SAM 低分辨率掩码 + IoU 预测。
        For training: DPG output + SAM low-res masks + IoU predictions.
    predict(): 推理用 (@no_grad), score = sigmoid(objectness) × iou_pred,
        score_thr 过滤 — 无 NMS、无 top-k。
        For inference: score-filtered masks — NO NMS, NO top-k.

v2 变更 | v2 changes:
    - prototype [256] → support_memory [M, 256]: SupportEncoder 输出替代 Mean Prototype
    - DPG 新增 support cross-attention + dense prompt generation
    - QueryMaskDecoder 接受 support-conditioned dense prompt (替代 no_mask_embed)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from adasam.decoder import QueryMaskDecoder, QueryMaskDecoderConfig
from adasam.prompt import DensePromptGenerator, DensePromptGeneratorConfig, DPGOutput
from adasam.prompt.coarse_prior import CoarsePriorModule
from adasam.support_encoder import SupportEncoder, SupportEncoderConfig


@dataclass(frozen=True)
class AdaSAMModelConfig:
    """组合模型配置 | composite model configuration."""

    dpg: DensePromptGeneratorConfig
    decoder: QueryMaskDecoderConfig
    support_encoder: SupportEncoderConfig
    use_coarse_prior: bool = True  # SAM-RSP 式粗先验模块 | coarse prior module

    @classmethod
    def from_dict(cls, cfg: dict) -> "AdaSAMModelConfig":
        """从完整 yaml 配置字典构建 | build from the full yaml config dict.

        读取 cfg["prompt_generator"], cfg["decoder"], cfg["support_encoder"] 段,
        缺省时用默认值。
        Reads the "prompt_generator", "decoder", "support_encoder" blocks;
        defaults if absent.
        """
        pg_cfg = cfg.get("prompt_generator", {})
        return cls(
            dpg=DensePromptGeneratorConfig.from_dict(pg_cfg),
            decoder=QueryMaskDecoderConfig.from_dict(cfg.get("decoder", {})),
            support_encoder=SupportEncoderConfig.from_dict(
                cfg.get("support_encoder", {})
            ),
            use_coarse_prior=bool(pg_cfg.get("use_coarse_prior", True)),
        )


class AdaSAMModel(nn.Module):
    """DPG + SupportEncoder + SAM 解码器组合 | DPG + SupportEncoder + SAM decoder composite.

    :param sam: MobileSAM Sam 实例 (取 prompt_encoder / mask_decoder) | Sam instance.
    :param cfg: :class:`AdaSAMModelConfig`.
    """

    def __init__(self, sam: nn.Module, cfg: AdaSAMModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.support_encoder = SupportEncoder(cfg.support_encoder)
        self.dpg = DensePromptGenerator(cfg.dpg)
        self.sam_decoder = QueryMaskDecoder(sam.prompt_encoder, sam.mask_decoder, cfg.decoder)

        # SAM-RSP 式粗先验模块 (opt-in, 默认启用)
        # SAM-RSP style coarse prior module (opt-in, enabled by default)
        if cfg.use_coarse_prior:
            self.coarse_prior = CoarsePriorModule(embed_dim=cfg.dpg.embed_dim)
        else:
            self.coarse_prior = None

    @property
    def num_queries(self) -> int:
        return self.cfg.dpg.num_queries

    def forward_train(
        self,
        query_features: torch.Tensor,
        support_features: torch.Tensor,
        support_masks: torch.Tensor,
    ) -> tuple[DPGOutput, torch.Tensor, torch.Tensor]:
        """训练前向 | Training forward.

        :param query_features: [1, C, gh, gw] CAT 适配后的查询图特征 | adapted features.
        :param support_features: [K, C, gh, gw] K 张 support 特征图 (已 CAT 适配).
        :param support_masks: [K, gh, gw] K 张 FG 掩码 (已 resize 到特征图尺寸).
        :return: (dpg_out, low_res_logits [N,1,256,256], iou_pred [N,1]).
        """
        # 1. Build support memory tokens
        support_memory = self.support_encoder(support_features, support_masks)  # [M, C]

        # 2. Coarse Prior (SAM-RSP style): enrich query features with RSP + pixel prototype
        if self.coarse_prior is not None:
            query_features, _rsp_map = self.coarse_prior(query_features, support_memory)

        # 3. DPG: generate instance queries + dense prompt (with support conditioning)
        dense_pe = self.sam_decoder.prompt_encoder.get_dense_pe()
        dpg_out = self.dpg(query_features, support_memory, dense_pe)

        # 4. Build dense prompt for SAM decoder:
        #    优先使用 support-conditioned dense prompt, 与 no_mask_embed 残差融合
        #    prefer support-conditioned dense prompt; residual with no_mask_embed
        if dpg_out.dense_prompt is not None:
            no_mask = (
                self.sam_decoder.prompt_encoder.no_mask_embed.weight
                .view(1, -1, 1, 1)
            )
            # residual: no_mask_embed + support_dense (zero-init → identity start)
            dense_override = no_mask + dpg_out.dense_prompt
        else:
            dense_override = None

        # 5. SAM decoder
        low_res, iou_pred = self.sam_decoder(
            query_features, dpg_out.instance_queries, dense_override
        )
        return dpg_out, low_res, iou_pred

    @torch.no_grad()
    def predict(
        self,
        query_features: torch.Tensor,
        support_features: torch.Tensor,
        support_masks: torch.Tensor,
        input_size: tuple[int, int],
        original_size: tuple[int, int],
        score_thr: float = 0.3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """推理 | Inference.

        :param query_features: [1, C, gh, gw] 查询图特征 | query features.
        :param support_features: [K, C, gh, gw] support 特征图.
        :param support_masks: [K, gh, gw] FG 掩码 (已 resize).
        :param input_size: 预处理后的有效输入尺寸 | valid input size after preprocess.
        :param original_size: 原图尺寸 (tile 分辨率) | original (tile) size.
        :param score_thr: score = sigmoid(objectness) × iou_pred 的过滤阈值 | threshold.
        :return: (masks [n, H, W] bool, scores [n] ∈ [0,1]), n ≤ num_queries。
        """
        dpg_out, low_res, iou_pred = self.forward_train(
            query_features, support_features, support_masks
        )

        scores = dpg_out.objectness_logits.sigmoid() * iou_pred[:, 0].clamp(0.0, 1.0)
        keep = scores >= score_thr                               # [N] bool
        if not keep.any():
            h, w = original_size
            empty = torch.zeros(0, h, w, dtype=torch.bool, device=low_res.device)
            return empty, scores[keep]

        logits = self.sam_decoder.upscale_logits(
            low_res[keep], input_size, original_size
        )                                                        # [n, H, W]
        return logits > self.sam_decoder.mask_threshold, scores[keep]
