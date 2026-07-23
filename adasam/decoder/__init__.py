"""adasam.decoder — 查询驱动的 SAM 掩码解码器 | Query-prompted SAM mask decoder."""

from adasam.decoder.sam_mask_decoder import (
    QueryMaskDecoder,
    QueryMaskDecoderConfig,
)

__all__ = ["QueryMaskDecoder", "QueryMaskDecoderConfig"]
