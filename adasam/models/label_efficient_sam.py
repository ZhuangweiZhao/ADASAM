"""Independent MobileSAM baseline for label-efficient semantic segmentation."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from adasam.adapters import MultiScaleCATAdapter
from adasam.backbone import LabelEfficientMobileSAMBackbone
from adasam.models.decoder import LightweightSemanticDecoder
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
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = (
            MultiScaleCATAdapter(bottleneck_ratio=adapter_ratio)
            if use_cat_adapter
            else nn.Identity()
        )
        self.use_cat_adapter = use_cat_adapter
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
        self.decoder = LightweightSemanticDecoder(
            num_classes,
            decoder_dim=decoder_dim,
            enable_prompt_fusion=prompt_version == "v1",
            enable_spatial_prompt_fusion=prompt_version in {"v2", "v3"},
            spatial_prompt_mode=prompt_fusion_mode,
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
        ).to(device)

    def train(self, mode: bool = True) -> "LabelEfficientSAM":
        super().train(mode)
        self.backbone.eval()
        return self

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        size = (self.backbone.img_size, self.backbone.img_size)
        image = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
        return (image * 255.0 - self.pixel_mean) / self.pixel_std

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"expected image [B,3,H,W], got {tuple(image.shape)}")
        output_size = tuple(image.shape[-2:])
        with torch.no_grad():
            features = self.backbone(self._preprocess(image))
        adapted = self.adapter(features)
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
        with torch.no_grad():
            features = self.backbone(self._preprocess(image))
        adapted = self.adapter(features)
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
        return logits, prompts, prototype_aux

    def parameter_counts(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
