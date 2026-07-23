"""
Support Representation Encoder | 支持表征编码器.
=================================================

将 K 张 support image 的特征 + 掩码编码为 support memory tokens,
替代原有的 Mean Prototype 压缩, 保留空间结构的 support 信息。
Encodes K support features + masks into support memory tokens,
replacing the legacy Mean Prototype compression.

Exports:
    SupportEncoderConfig: configuration dataclass.
    SupportEncoder: the core encoder module.
"""

from adasam.support_encoder.support_encoder import (
    SupportEncoder,
    SupportEncoderConfig,
)

__all__ = ["SupportEncoder", "SupportEncoderConfig"]
