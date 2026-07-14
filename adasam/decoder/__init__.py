"""adasam.decoder — 原型驱动的 SAM 掩码解码器 | Prototype-prompted SAM mask decoder."""

from adasam.decoder.mask_decoder import (
    PromptMaskDecoder,
    PrototypeAdapter,
    InstanceMasks,
)
from adasam.decoder.prompt_generator import PromptGenerator

__all__ = ["PromptMaskDecoder", "PrototypeAdapter", "InstanceMasks", "PromptGenerator"]
