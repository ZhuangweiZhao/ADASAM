"""
AdaSAM 组合模型 | AdaSAM composite model.
===========================================

SemanticPriorGenerator + GeometricPrior + PromptFusion + SupportEncoder +
SemanticMaskDecoder — 双支路语义先验架构。

Dual-branch semantic prior architecture: Geometric Prior (support-query
similarity) + Semantic Prior (learned SPG) → PromptFusion → SAM Decoder.

前向 | Forward:
    forward_train(): 训练用, 返回 SPG 输出 + SAM 低分辨率掩码 + IoU 预测。
        For training: SPG output + SAM low-res masks + IoU predictions.
    predict(): 推理用 (@no_grad), 输出经过滤的二值掩码。
        For inference: score-filtered binary masks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from adasam.decoder import SemanticMaskDecoder, SemanticMaskDecoderConfig
from adasam.prompt import (
    GeometricPriorModule,
    PromptFusion,
    SemanticPriorGenerator,
    SemanticPriorGeneratorConfig,
    SPGOutput,
)
from adasam.support_encoder import SupportEncoder, SupportEncoderConfig
from adasam.utils.debug_trace import tracer


@dataclass(frozen=True)
class AdaSAMModelConfig:
    """组合模型配置 | Composite model configuration."""

    spg: SemanticPriorGeneratorConfig
    decoder: SemanticMaskDecoderConfig
    support_encoder: SupportEncoderConfig
    use_geometric_prior: bool = True
    use_prompt_fusion: bool = True
    fusion_mode: str = "concat"

    @classmethod
    def from_dict(cls, cfg: dict) -> "AdaSAMModelConfig":
        """从完整 yaml 配置字典构建 | Build from the full yaml config dict."""
        sp_cfg = cfg.get("semantic_prior", {})
        gp_cfg = cfg.get("geometric_prior", {})
        pf_cfg = cfg.get("prompt_fusion", {})
        return cls(
            spg=SemanticPriorGeneratorConfig.from_dict(sp_cfg),
            decoder=SemanticMaskDecoderConfig.from_dict(cfg.get("decoder", {})),
            support_encoder=SupportEncoderConfig.from_dict(
                cfg.get("support_encoder", {})
            ),
            use_geometric_prior=bool(gp_cfg.get("enabled", True)),
            use_prompt_fusion=bool(pf_cfg.get("enabled", True)),
            fusion_mode=str(pf_cfg.get("mode", "concat")),
        )


class AdaSAMModel(nn.Module):
    """SPG + GeometricPrior + PromptFusion + SupportEncoder + SAM Decoder.

    :param sam: MobileSAM Sam instance (取 prompt_encoder / mask_decoder).
    :param cfg: :class:`AdaSAMModelConfig`.
    """

    def __init__(self, sam: nn.Module, cfg: AdaSAMModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        embed_dim = cfg.spg.embed_dim

        self.support_encoder = SupportEncoder(cfg.support_encoder)
        self.spg = SemanticPriorGenerator(cfg.spg)
        self.sam_decoder = SemanticMaskDecoder(sam.prompt_encoder, sam.mask_decoder, cfg.decoder)

        # ── Geometric Prior (双支路: geometry branch) ──
        if cfg.use_geometric_prior:
            self.geometric_prior = GeometricPriorModule(embed_dim=embed_dim)
        else:
            self.geometric_prior = None

        # ── Prompt Fusion (双支路融合 → dense_prompt + sparse_token) ──
        if cfg.use_prompt_fusion and cfg.use_geometric_prior:
            self.prompt_fusion = PromptFusion(
                embed_dim=embed_dim, mode=cfg.fusion_mode
            )
        else:
            self.prompt_fusion = None

        # ── Dense prompt generator (fallback when PromptFusion disabled) ──
        # 从 support spatial features 生成 dense prompt, 原属 SPG 内部模块,
        # 提取到此以保证 SPG 职责单一 (只生成 semantic prior)。
        # Extracted from SPG to keep SPG's responsibility single (semantic prior only).
        self.spatial_prompt_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
        )
        nn.init.xavier_uniform_(self.spatial_prompt_proj[-1].weight, gain=1.0)
        nn.init.zeros_(self.spatial_prompt_proj[-1].bias)
        self.spatial_prompt_scale = nn.Parameter(torch.tensor(1.0))

        # Legacy fallback: global dense prompt from support_memory (无空间特征时)
        self.dense_pool_attn = nn.Linear(embed_dim, 1)
        self.dense_prompt_gen = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        nn.init.xavier_uniform_(self.dense_prompt_gen[-1].weight, gain=1.0)
        nn.init.zeros_(self.dense_prompt_gen[-1].bias)

    @property
    def num_probes(self) -> int:
        """内部语义探针数 (实现细节) | Internal semantic probe count."""
        return self.cfg.spg.num_probes

    # ── Dense prompt generation (extracted from SPG) ──

    def _build_dense_prompt(
        self,
        support_memory: torch.Tensor,
        support_features: torch.Tensor | None = None,
        support_masks_grid: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """从 support 生成 dense prompt | Build dense prompt from support.

        优先使用空间通路 (masked mean of support features),
        回退到全局通路 (attention-pooled support memory)。

        Prefers spatial path (masked mean of support features),
        falls back to global path (attention-pooled support memory).

        :param support_memory: [M, C] support memory tokens.
        :param support_features: [K, C, gh, gw] optional spatial support features.
        :param support_masks_grid: [K, gh, gw] optional support FG masks.
        :return: dense_prompt [1, C, gh, gw] or None.
        """
        has_support = support_memory.shape[0] > 0
        if not has_support:
            return None

        if support_features is not None and support_masks_grid is not None:
            # Spatial path: masked mean → projection
            masked = support_features * support_masks_grid.unsqueeze(1)
            support_spatial = masked.mean(dim=0, keepdim=True)  # [1, C, gh, gw]
            dense = self.spatial_prompt_proj(support_spatial)
            dense = self.spatial_prompt_scale * dense
            return dense  # [1, C, gh, gw]
        else:
            # Legacy global path: attention-pooled memory → MLP
            attn_scores = self.dense_pool_attn(support_memory)
            attn_weights = torch.softmax(attn_scores, dim=0)
            support_summary = (support_memory * attn_weights).sum(dim=0)
            dense_mod = self.dense_prompt_gen(support_summary)
            return dense_mod.view(1, -1, 1, 1)  # [1, C, 1, 1]

    # ── Training forward ──

    def forward_train(
        self,
        query_features: torch.Tensor,
        support_features: torch.Tensor,
        support_masks: torch.Tensor,
    ) -> tuple[SPGOutput, torch.Tensor, torch.Tensor]:
        """训练前向 | Training forward.

        :param query_features: [1, C, gh, gw] CAT-adapted query features.
        :param support_features: [K, C, gh, gw] K support features (CAT-adapted).
        :param support_masks: [K, gh, gw] K FG masks (resized to feature grid).
        :return: (spg_out, low_res_logits [1,1,256,256], iou_pred [1,1]).
        """
        # 0. Trace inputs
        tracer.section("AdaSAM.forward_train — Inputs")
        tracer.tensor("query_features", query_features, spatial=True)
        tracer.tensor("support_features", support_features, spatial=True)
        tracer.tensor("support_masks", support_masks)

        # 1. Support Encoder → support memory [M, C]
        support_memory = self.support_encoder(support_features, support_masks)
        tracer.tensor("support_memory", support_memory, detail=True)

        # 2. Geometric Prior: support-query similarity → geometric_prior [1,C,H,W]
        if self.geometric_prior is not None:
            geometric_prior = self.geometric_prior(query_features, support_memory)
            tracer.section("AdaSAM.forward_train — GeometricPrior")
            tracer.tensor("geometric_prior", geometric_prior, spatial=True)
        else:
            geometric_prior = None

        # 3. SPG: query_features + support_memory → semantic_prior + prior_mask
        #    SPG 不再接收 support_features/support_masks_grid,
        #    也不再生产 dense_prompt/sparse_token。
        dense_pe = self.sam_decoder.prompt_encoder.get_dense_pe()
        spg_out = self.spg(query_features, support_memory, dense_pe)

        tracer.section("AdaSAM.forward_train — SPG Output")
        tracer.tensor("semantic_prior", spg_out.semantic_prior, spatial=True)
        tracer.tensor("prior_mask", spg_out.prior_mask)
        if spg_out.prior_aux:
            aux0 = spg_out.prior_aux[0]
            tracer.tensor("prior_aux[0].prior_mask", aux0["prior_mask"].unsqueeze(0))

        # 4. PromptFusion → dense_prompt + sparse_token (唯一来源)
        if self.prompt_fusion is not None and geometric_prior is not None:
            dense_prompt, sparse_token = self.prompt_fusion(
                geometric_prior, spg_out.semantic_prior
            )
            tracer.section("AdaSAM.forward_train — PromptFusion")
            tracer.tensor("fused_dense_prompt", dense_prompt, spatial=True)
            tracer.tensor("fused_sparse_token", sparse_token.unsqueeze(0))
        else:
            # Fallback: SPG's semantic_prior serves as dense_prompt,
            # spatial mean pool gives sparse_token
            dense_prompt = self._build_dense_prompt(
                support_memory, support_features, support_masks
            )
            if dense_prompt is None:
                # Ultimate fallback: use semantic_prior directly
                dense_prompt = spg_out.semantic_prior

            sparse_token = (
                spg_out.semantic_prior if dense_prompt is None
                else dense_prompt
            ).mean(dim=(2, 3))  # [1, C]

        # 5. SAM Decoder: refine prior → fine mask
        low_res, iou_pred = self.sam_decoder(
            query_features, sparse_token, dense_prompt
        )
        tracer.section("AdaSAM.forward_train — SAM Decoder Output")
        tracer.tensor("low_res_masks [1,1,256,256]", low_res)
        tracer.tensor("iou_pred [1,1]", iou_pred)

        return spg_out, low_res, iou_pred

    # ── Inference ──

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

        :param query_features: [1, C, gh, gw] query features.
        :param support_features: [K, C, gh, gw] support features.
        :param support_masks: [K, gh, gw] FG masks (resized).
        :param input_size: valid input size after preprocess.
        :param original_size: original tile size.
        :param score_thr: confidence threshold (sigmoid × iou_pred).
        :return: (masks [1, H, W] bool, scores [1] ∈ [0,1]).
        """
        spg_out, low_res, iou_pred = self.forward_train(
            query_features, support_features, support_masks
        )

        # Single output: confidence = iou_pred only
        score = iou_pred[0, 0].clamp(0.0, 1.0)  # scalar
        if score < score_thr:
            h, w = original_size
            empty = torch.zeros(1, h, w, dtype=torch.bool, device=low_res.device)
            return empty, torch.tensor([score], device=low_res.device)

        logits = self.sam_decoder.upscale_logits(
            low_res, input_size, original_size
        )  # [1, H, W]
        return logits > self.sam_decoder.mask_threshold, torch.tensor([score], device=low_res.device)
