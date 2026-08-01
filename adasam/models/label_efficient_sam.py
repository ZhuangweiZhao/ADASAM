"""Independent MobileSAM baseline for label-efficient semantic segmentation."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from adasam.adapters import MultiScaleCATAdapter
from adasam.backbone import LabelEfficientMobileSAMBackbone
from adasam.models.decoder import LightweightSemanticDecoder
from adasam.models.prompt import DefectPromptGenerator
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
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = MultiScaleCATAdapter(bottleneck_ratio=adapter_ratio)
        self.use_dapg = use_dapg
        self.prompt_generator = (
            DefectPromptGenerator(num_prompt=num_prompt) if use_dapg else None
        )
        self.decoder = LightweightSemanticDecoder(
            num_classes, decoder_dim=decoder_dim, enable_prompt_fusion=use_dapg
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
    ) -> "LabelEfficientSAM":
        backbone = LabelEfficientMobileSAMBackbone.build(
            checkpoint, model_type=model_type, device=device, img_size=img_size
        )
        return cls(
            backbone, num_classes, decoder_dim, adapter_ratio,
            use_dapg=use_dapg, num_prompt=num_prompt,
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
        prompt_tokens = self.prompt_generator(adapted) if self.prompt_generator is not None else None
        return self.decoder(adapted, output_size=output_size, prompt_tokens=prompt_tokens)

    def forward_with_prompts(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Diagnostic forward returning logits and generated prompt tokens."""
        output_size = tuple(image.shape[-2:])
        with torch.no_grad():
            features = self.backbone(self._preprocess(image))
        adapted = self.adapter(features)
        prompts = self.prompt_generator(adapted) if self.prompt_generator is not None else None
        return self.decoder(adapted, output_size, prompt_tokens=prompts), prompts

    def parameter_counts(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
