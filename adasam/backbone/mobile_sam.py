"""
MobileSAM 骨干 | MobileSAM backbone.
====================================

AdaSAM 的唯一骨干: MobileSAM 图像编码器 (TinyViT)。职责单一 —— 把预处理后的图像
编码为 256-d 图像嵌入, 供原型构建与提示式掩码解码使用。
The single backbone of AdaSAM: the MobileSAM image encoder (TinyViT). Single
responsibility — encode a preprocessed image into a 256-d image embedding consumed
by prototype building and prompt-based mask decoding.

契约 | Contract::

    forward(image[B, 3, 1024, 1024]) -> {"image_embedding": [B, 256, 64, 64]}

设计要点 | Design notes:
    - 骨干**始终冻结且处于 eval** (frozen feature extractor)。参数 requires_grad=False,
      并覆写 train() 使其忽略训练模式, 避免误更新 BN/统计量。
      The backbone is ALWAYS frozen and in eval. Params requires_grad=False, and train()
      is overridden to ignore the requested mode (no accidental BN/statistic updates).
    - 只拥有 image_encoder。prompt_encoder / mask_decoder 由 build_mobile_sam 返回的同一
      Sam 提供, 交给 adasam.decoder 管理 (那部分可训练) —— 各模块单一职责。
      Owns only the image_encoder. prompt_encoder / mask_decoder come from the same Sam
      (via build_mobile_sam) and are managed by adasam.decoder (the trainable part).
    - vendored MobileSAM 通过 sys.path 注入, 不作为 pip 包安装 (镜像 frozen-third-party 模式)。
      Vendored MobileSAM is injected via sys.path, not pip-installed (frozen-third-party pattern).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# vendored MobileSAM 根目录 | vendored MobileSAM root: <repo>/thirdparty/MobileSAM
_MOBILE_SAM_ROOT = Path(__file__).resolve().parents[2] / "thirdparty" / "MobileSAM"


def _ensure_mobile_sam_on_path() -> None:
    """将 vendored MobileSAM 注入 sys.path | Inject vendored MobileSAM into sys.path (idempotent)."""
    p = str(_MOBILE_SAM_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


def build_mobile_sam(
    checkpoint: str | Path,
    model_type: str = "vit_t",
    device: str | torch.device = "cpu",
):
    """构建完整 MobileSAM (Sam) 模型 | Build the full MobileSAM (Sam) model.

    返回的 Sam 捆绑三部分 (image_encoder / prompt_encoder / mask_decoder), 共享 256-d 空间。
    模型装配处 (trainer) 调用一次, 把三部分分发给 backbone 与 decoder, 避免重复加载。
    The returned Sam bundles the three parts sharing the 256-d space. The model-assembly
    site (trainer) calls this once and distributes the parts to backbone and decoder.

    :param checkpoint: mobile_sam.pt 权重路径 | path to mobile_sam.pt.
    :param model_type: MobileSAM 变体, 固定 "vit_t" | MobileSAM variant, fixed "vit_t".
    :param device: 目标设备 | target device.
    :return: mobile_sam.modeling.Sam (已 eval) | Sam instance (in eval mode).
    """
    _ensure_mobile_sam_on_path()
    from mobile_sam import sam_model_registry  # vendored

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device)
    sam.eval()
    return sam


class MobileSAMBackbone(nn.Module):
    """MobileSAM 图像编码器包装 (冻结) | Frozen wrapper around the MobileSAM image encoder."""

    def __init__(self, image_encoder: nn.Module, img_size: int = 1024) -> None:
        """
        :param image_encoder: MobileSAM TinyViT 编码器 | the MobileSAM TinyViT encoder.
        :param img_size: 编码器输入边长 | encoder input side length (1024).
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.img_size = int(img_size)
        self._freeze()

    @classmethod
    def build(
        cls,
        checkpoint: str | Path,
        model_type: str = "vit_t",
        device: str | torch.device = "cpu",
    ) -> "MobileSAMBackbone":
        """从权重直接构建 (独立使用/测试) | Build directly from a checkpoint (standalone/testing).

        注意: 会构建完整 Sam 但只保留 image_encoder。模型装配时应改用 build_mobile_sam
        以便与 decoder 共享同一 Sam。
        Note: builds the full Sam but keeps only the image_encoder. In model assembly,
        prefer build_mobile_sam to share one Sam with the decoder.
        """
        sam = build_mobile_sam(checkpoint, model_type=model_type, device=device)
        return cls(sam.image_encoder, img_size=sam.image_encoder.img_size)

    def _freeze(self) -> None:
        """冻结所有参数并置于 eval | Freeze all params and force eval."""
        for p in self.image_encoder.parameters():
            p.requires_grad_(False)
        self.image_encoder.eval()

    def train(self, mode: bool = True) -> "MobileSAMBackbone":
        """覆写: 冻结骨干始终保持 eval, 忽略请求的 mode | Frozen backbone: always eval, ignore mode."""
        super().train(False)
        self.image_encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """图像 → 图像嵌入 | Image → image embedding.

        :param image: 已预处理张量 [B, 3, 1024, 1024] | preprocessed tensor.
        :return: {"image_embedding": [B, 256, 64, 64]}.
        """
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"expected [B, 3, H, W], got {tuple(image.shape)}")
        embedding = self.image_encoder(image)
        return {"image_embedding": embedding}
