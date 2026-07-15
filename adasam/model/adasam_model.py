"""
AdaSAM 组合模型 | AdaSAM composite model.
===========================================

DensePromptGenerator + QueryMaskDecoder 的组合模块 — trainer 与 evaluator 共用
同一构建路径, checkpoint "model" 键即本模块的 state_dict。
Composite of DensePromptGenerator + QueryMaskDecoder — the single construction
path shared by trainer and evaluator; the checkpoint "model" key is this
module's state_dict.

前向 | Forward:
    forward_train(): 训练用, 返回 DPG 输出 + SAM 低分辨率掩码 + IoU 预测。
        For training: DPG output + SAM low-res masks + IoU predictions.
    predict(): 推理用 (@no_grad), score = sigmoid(objectness) × iou_pred,
        score_thr 过滤 — 无 NMS、无 top-k。
        For inference: score-filtered masks — NO NMS, NO top-k.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from adasam.decoder import QueryMaskDecoder, QueryMaskDecoderConfig
from adasam.prompt import DensePromptGenerator, DensePromptGeneratorConfig, DPGOutput


@dataclass(frozen=True)
class AdaSAMModelConfig:
    """组合模型配置 | composite model configuration."""

    dpg: DensePromptGeneratorConfig
    decoder: QueryMaskDecoderConfig

    @classmethod
    def from_dict(cls, cfg: dict) -> "AdaSAMModelConfig":
        """从完整 yaml 配置字典构建 | build from the full yaml config dict.

        读取 cfg["prompt_generator"] 与 cfg["decoder"] 段, 缺省时用默认值。
        Reads the "prompt_generator" and "decoder" blocks; defaults if absent.
        """
        return cls(
            dpg=DensePromptGeneratorConfig.from_dict(cfg.get("prompt_generator", {})),
            decoder=QueryMaskDecoderConfig.from_dict(cfg.get("decoder", {})),
        )


class AdaSAMModel(nn.Module):
    """DPG + SAM 解码器组合 | DPG + SAM decoder composite.

    :param sam: MobileSAM Sam 实例 (取 prompt_encoder / mask_decoder) | Sam instance.
    :param cfg: :class:`AdaSAMModelConfig`.
    """

    def __init__(self, sam: nn.Module, cfg: AdaSAMModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.dpg = DensePromptGenerator(cfg.dpg)
        self.sam_decoder = QueryMaskDecoder(sam.prompt_encoder, sam.mask_decoder, cfg.decoder)

    @property
    def num_queries(self) -> int:
        return self.cfg.dpg.num_queries

    def forward_train(
        self,
        query_features: torch.Tensor,
        prototype: torch.Tensor,
    ) -> tuple[DPGOutput, torch.Tensor, torch.Tensor]:
        """训练前向 | Training forward.

        :param query_features: [1, C, gh, gw] CAT 适配后的查询图特征 | adapted features.
        :param prototype: [C] 类原型 (语义条件) | class prototype (semantic condition).
        :return: (dpg_out, low_res_logits [N,1,256,256], iou_pred [N,1]).
        """
        dense_pe = self.sam_decoder.prompt_encoder.get_dense_pe()
        dpg_out = self.dpg(query_features, prototype, dense_pe)
        low_res, iou_pred = self.sam_decoder(
            query_features, dpg_out.instance_queries, prototype
        )
        return dpg_out, low_res, iou_pred

    @torch.no_grad()
    def predict(
        self,
        query_features: torch.Tensor,
        prototype: torch.Tensor,
        input_size: tuple[int, int],
        original_size: tuple[int, int],
        score_thr: float = 0.3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """推理 | Inference.

        :param query_features: [1, C, gh, gw] 查询图特征 | query features.
        :param prototype: [C] 类原型 | class prototype.
        :param input_size: 预处理后的有效输入尺寸 | valid input size after preprocess.
        :param original_size: 原图尺寸 (tile 分辨率) | original (tile) size.
        :param score_thr: score = sigmoid(objectness) × iou_pred 的过滤阈值 | threshold.
        :return: (masks [n, H, W] bool, scores [n] ∈ [0,1]), n ≤ num_queries。
        """
        dpg_out, low_res, iou_pred = self.forward_train(query_features, prototype)

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
