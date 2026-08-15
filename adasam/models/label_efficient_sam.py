"""Independent MobileSAM baseline for label-efficient semantic segmentation."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from adasam.adapters import MultiScaleCATAdapter
from adasam.backbone import LabelEfficientMobileSAMBackbone
from adasam.models.decoder import BoundaryAwareSemanticDecoder, LightweightSemanticDecoder
from adasam.models.prompt import (
    DefectAwarePromptGeneratorV2,
    DefectPromptGenerator,
    FrequencyAwareDefectPromptGenerator,
)
from adasam.models.prototype import DefectPrototypeMemory
from adasam.utils.transforms import PIXEL_MEAN, PIXEL_STD


class LabelEfficientSAM(nn.Module):
    """Frozen MobileSAM + multi-scale CAT adapter + lightweight semantic decoder."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        decoder_dim: int = 96,
        adapter_ratio: float = 0.25,
        use_dapg: bool = False,
        num_prompt: int = 16,
        prompt_version: str | None = None,
        prompt_fusion_mode: str = "both",
        use_cat_adapter: bool = True,
        prototype_version: str = "none",
        prototype_momentum: float = 0.9,
        decoder_version: str = "lightweight",
        feature_scales: str = "p3_p4_embedding",
        adapter_placement: str = "pre_fusion",
        fusion_version: str = "hierarchical",
        representation_budget: int = 3,
        spatial_policy: str = "adaptive",
        feature_retention_ratio: float = 1.0,
        spatial_budget_temperature: float = 1.0,
        static_importance_map: torch.Tensor | None = None,
        use_input_adapter: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.use_input_adapter = use_input_adapter
        self.encoder_requires_grad = False
        self.input_adapter = nn.Conv2d(3, 3, 1, bias=True) if use_input_adapter else nn.Identity()
        if use_input_adapter:
            with torch.no_grad():
                self.input_adapter.weight.copy_(torch.eye(3).view(3, 3, 1, 1))
                self.input_adapter.bias.zero_()
        scale_names = {
            "p3": ("P3",), "p4": ("P4",), "embedding": ("embedding",),
            "p3_p4": ("P3", "P4"), "p3_embedding": ("P3", "embedding"),
            "p4_embedding": ("P4", "embedding"),
            "p3_p4_embedding": ("P3", "P4", "embedding"),
        }
        if feature_scales not in scale_names:
            raise ValueError(
                "feature_scales must be one of: p3, p4, embedding, p3_p4, p3_embedding, p4_embedding, p3_p4_embedding"
            )
        feature_dims = {"P3": 128, "P4": 160, "embedding": 256}
        selected_dims = {name: feature_dims[name] for name in scale_names[feature_scales]}
        if adapter_placement not in {"pre_fusion", "post_fusion"}:
            raise ValueError("adapter_placement must be one of: pre_fusion, post_fusion")
        self.feature_scales = feature_scales
        self.fusion_version = fusion_version
        self.adapter_placement = adapter_placement
        use_pre_fusion_adapter = use_cat_adapter and adapter_placement == "pre_fusion"
        self.adapter = (
            MultiScaleCATAdapter(feature_dims=selected_dims, bottleneck_ratio=adapter_ratio)
            if use_pre_fusion_adapter
            else nn.Identity()
        )
        self.use_cat_adapter = use_cat_adapter
        self.use_pre_fusion_adapter = use_pre_fusion_adapter
        if prototype_version not in {"none", "dpm"}:
            raise ValueError("prototype_version must be one of: none, dpm")
        self.prototype_version = prototype_version
        self.prototype_memory = (
            DefectPrototypeMemory(num_classes, feature_dim=256, momentum=prototype_momentum)
            if prototype_version == "dpm"
            else None
        )
        if prompt_version is None:
            prompt_version = "v1" if use_dapg else "none"
        if prompt_version not in {"none", "v1", "v2", "v3"}:
            raise ValueError("prompt_version must be one of: none, v1, v2, v3")
        self.prompt_version = prompt_version
        self.use_dapg = prompt_version == "v1"
        self.prompt_generator = (
            DefectPromptGenerator(num_prompt=num_prompt) if prompt_version == "v1"
            else DefectAwarePromptGeneratorV2(num_prompt=num_prompt) if prompt_version == "v2"
            else FrequencyAwareDefectPromptGenerator(num_prompt=num_prompt) if prompt_version == "v3"
            else None
        )
        if decoder_version not in {"lightweight", "boundary_aux", "boundary"}:
            raise ValueError("decoder_version must be one of: lightweight, boundary_aux, boundary")
        self.decoder_version = decoder_version
        decoder_class = (
            LightweightSemanticDecoder
            if decoder_version == "lightweight"
            else BoundaryAwareSemanticDecoder
        )
        decoder_kwargs = {}
        if decoder_version != "lightweight":
            decoder_kwargs["enable_boundary_fusion"] = decoder_version == "boundary"
        self.decoder = decoder_class(
            num_classes,
            decoder_dim=decoder_dim,
            enable_prompt_fusion=prompt_version == "v1",
            enable_spatial_prompt_fusion=prompt_version in {"v2", "v3"},
            spatial_prompt_mode=prompt_fusion_mode,
            feature_scales=feature_scales,
            fusion_version=fusion_version,
            representation_budget=representation_budget,
            spatial_policy=spatial_policy,
            feature_retention_ratio=feature_retention_ratio,
            spatial_budget_temperature=spatial_budget_temperature,
            static_importance_map=static_importance_map,
            post_fusion_adapter=use_cat_adapter and adapter_placement == "post_fusion",
            adapter_ratio=adapter_ratio,
            **decoder_kwargs,
        )
        self.register_buffer(
            "pixel_mean", torch.tensor(PIXEL_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "pixel_std", torch.tensor(PIXEL_STD).view(1, 3, 1, 1), persistent=False
        )

    @classmethod
    def build(
        cls,
        checkpoint: str | Path,
        num_classes: int,
        img_size: int = 224,
        model_type: str = "vit_t",
        device: str | torch.device = "cpu",
        decoder_dim: int = 96,
        adapter_ratio: float = 0.25,
        use_dapg: bool = False,
        num_prompt: int = 16,
        prompt_version: str | None = None,
        prompt_fusion_mode: str = "both",
        use_cat_adapter: bool = True,
        prototype_version: str = "none",
        prototype_momentum: float = 0.9,
        decoder_version: str = "lightweight",
        feature_scales: str = "p3_p4_embedding",
        adapter_placement: str = "pre_fusion",
        fusion_version: str = "hierarchical",
        representation_budget: int = 3,
        spatial_policy: str = "adaptive",
        feature_retention_ratio: float = 1.0,
        spatial_budget_temperature: float = 1.0,
        static_importance_map: torch.Tensor | None = None,
        use_input_adapter: bool = False,
    ) -> "LabelEfficientSAM":
        backbone = LabelEfficientMobileSAMBackbone.build(
            checkpoint, model_type=model_type, device=device, img_size=img_size
        )
        return cls(
            backbone, num_classes, decoder_dim, adapter_ratio,
            use_dapg=use_dapg, num_prompt=num_prompt,
            prompt_version=prompt_version,
            prompt_fusion_mode=prompt_fusion_mode,
            use_cat_adapter=use_cat_adapter,
            prototype_version=prototype_version,
            prototype_momentum=prototype_momentum,
            decoder_version=decoder_version,
            feature_scales=feature_scales,
            adapter_placement=adapter_placement,
            fusion_version=fusion_version,
            representation_budget=representation_budget,
            spatial_policy=spatial_policy,
            feature_retention_ratio=feature_retention_ratio,
            spatial_budget_temperature=spatial_budget_temperature,
            static_importance_map=static_importance_map,
            use_input_adapter=use_input_adapter,
        ).to(device)

    def train(self, mode: bool = True) -> "LabelEfficientSAM":
        super().train(mode)
        if self.encoder_requires_grad:
            self.backbone.train(mode)
        else:
            self.backbone.eval()
        return self

    def set_encoder_trainable(self, trainable: bool = True) -> None:
        """Switch between the frozen protocol and full TinyViT fine-tuning."""
        self.encoder_requires_grad = bool(trainable)
        self.backbone.set_encoder_trainable(self.encoder_requires_grad)
        self.backbone.train(self.training) if self.encoder_requires_grad else self.backbone.eval()

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        size = (self.backbone.img_size, self.backbone.img_size)
        image = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
        image = self.input_adapter(image)
        return (image * 255.0 - self.pixel_mean) / self.pixel_std

    def _adapt_features(
        self, features: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        adapted = self.adapter(features)
        return {**features, **adapted} if self.use_pre_fusion_adapter else adapted

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"expected image [B,3,H,W], got {tuple(image.shape)}")
        output_size = tuple(image.shape[-2:])
        # Keep the encoder frozen while allowing the small input adapter to learn.
        preprocessed = self._preprocess(image)
        # Parameters remain frozen, but the adapter needs a gradient through
        # the encoder with respect to its input.
        if self.use_input_adapter or self.encoder_requires_grad:
            features = self.backbone(preprocessed)
        else:
            with torch.no_grad():
                features = self.backbone(preprocessed)
        adapted = self._adapt_features(features)
        prompts = self.prompt_generator(adapted) if self.prompt_generator is not None else None
        decoder_features = adapted
        if self.prototype_memory is not None:
            enhanced, _ = self.prototype_memory(adapted["embedding"], update_memory=False)
            decoder_features = {**adapted, "embedding": enhanced}
        return self.decoder(
            decoder_features, output_size=output_size,
            prompt_tokens=prompts if self.prompt_version == "v1" else None,
            prompt=prompts if self.prompt_version in {"v2", "v3"} else None,
        )

    def forward_with_prompts(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor] | None]:
        """Diagnostic forward returning logits and the selected prompt representation."""
        logits, prompts, _ = self.forward_with_auxiliary(image)
        return logits, prompts

    def forward_with_auxiliary(
        self, image: torch.Tensor, target: torch.Tensor | None = None
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | dict[str, torch.Tensor] | None,
        dict[str, torch.Tensor] | None,
    ]:
        """Training/diagnostic forward exposing prompt and prototype representations."""
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"expected image [B,3,H,W], got {tuple(image.shape)}")
        output_size = tuple(image.shape[-2:])
        if self.encoder_requires_grad:
            features = self.backbone(self._preprocess(image))
        else:
            with torch.no_grad():
                features = self.backbone(self._preprocess(image))
        adapted = self._adapt_features(features)
        prompts = self.prompt_generator(adapted) if self.prompt_generator is not None else None
        prototype_aux = None
        decoder_features = adapted
        if self.prototype_memory is not None:
            enhanced, prototype_aux = self.prototype_memory(adapted["embedding"], target=target)
            decoder_features = {**adapted, "embedding": enhanced}
        logits = self.decoder(
            decoder_features,
            output_size,
            prompt_tokens=prompts if self.prompt_version == "v1" else None,
            prompt=prompts if self.prompt_version in {"v2", "v3"} else None,
        )
        auxiliary = prototype_aux or {}
        boundary_logits = getattr(self.decoder, "last_boundary_logits", None)
        if boundary_logits is not None:
            auxiliary["boundary_logits"] = boundary_logits
        return logits, prompts, auxiliary or None

    def parameter_counts(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
