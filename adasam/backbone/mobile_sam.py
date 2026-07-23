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


def build_mobile_sam_for_size(
    checkpoint: str | Path,
    target_size: int = 224,
    model_type: str = "vit_t",
    device: str | torch.device = "cpu",
):
    """构建适配特定输入尺寸的 TinyViT 编码器 | Build TinyViT encoder adapted to input size.

    MobileSAM 官方权重在 img_size=1024 上训练, 但 TinyViT 的卷积/注意力权重与空间尺寸无关。
    此函数用目标尺寸重建 TinyViT 并加载预训练权重, 适用于非 1024 的输入 (如 NEU_Seg 200→224)。

    :param checkpoint: mobile_sam.pt 权重路径.
    :param target_size: 目标输入边长 (如 224). 必须是 32 的倍数且满足 window_size 整除约束.
    :param model_type: "vit_t" (TinyViT-5M).
    :param device: 目标设备.
    :return: TinyViT image_encoder (eval mode, 已加载权重).
    """
    _ensure_mobile_sam_on_path()
    from mobile_sam.modeling.tiny_vit_sam import TinyViT

    ckpt_path = str(checkpoint)

    # 用目标尺寸重建 TinyViT-5M
    encoder = TinyViT(
        img_size=target_size, in_chans=3, num_classes=1000,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4., drop_rate=0., drop_path_rate=0.0,
        use_checkpoint=False, mbconv_expand_ratio=4.0, local_conv_size=3,
    )

    # 从 MobileSAM checkpoint 加载 image_encoder 权重
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # 兼容不同保存格式: 有时直接是 encoder state, 有时包裹在 Sam 中
    if "image_encoder" in state_dict:
        ie_state = state_dict["image_encoder"]
    elif "image_encoder.patch_embed.seq.c.weight" in state_dict:
        ie_state = state_dict
    else:
        # 尝试匹配前缀
        ie_state = {k.replace("image_encoder.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("image_encoder.")}
    if not ie_state:
        raise RuntimeError(f"Cannot extract image_encoder weights from {ckpt_path}")

    # 加载权重 (strict=False 容忍 buffer 差异如 attention_bias_idxs)
    missing, unexpected = encoder.load_state_dict(ie_state, strict=False)
    if missing:
        # 只警告 shape-mismatch 的 key, buffer 差异可忽略
        real_missing = [k for k in missing if not k.endswith("attention_bias_idxs")]
        if real_missing:
            print(f"  [WARN] Missing keys: {real_missing}")
    encoder.to(device)
    encoder.eval()
    return encoder


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


class MultiScaleMobileSAMBackbone(nn.Module):
    """多尺度 MobileSAM 骨干 | Multi-scale MobileSAM backbone.

    与 MobileSAMBackbone 同源 (TinyViT-5M, SA-1B 预训练), 但输出全部 4 个 stage 的特征。
    Same encoder as MobileSAMBackbone, but returns all 4 stage features for multi-scale decoding.

    契约 | Contract::

        forward(image[B, 3, 1024, 1024]) -> {
            "stage0": [B,  64, 256, 256],   # H/4  — low-level edges/texture
            "stage1": [B, 128, 128, 128],   # H/8  — mid-level patterns
            "stage2": [B, 160,  64,  64],   # H/16 — high-level parts
            "stage3": [B, 256,  64,  64],   # H/16 — SAM-aligned neck output
        }
    """

    def __init__(self, image_encoder: nn.Module, img_size: int = 1024) -> None:
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
        img_size: int | None = None,
    ) -> "MultiScaleMobileSAMBackbone":
        """Build backbone with optional custom input size.

        :param img_size: If None, uses default 1024. Use 224 for NEU_Seg native resolution.
        """
        if img_size is not None and img_size != 1024:
            encoder = build_mobile_sam_for_size(
                checkpoint, target_size=img_size, model_type=model_type, device=device
            )
        else:
            sam = build_mobile_sam(checkpoint, model_type=model_type, device=device)
            encoder = sam.image_encoder
        return cls(encoder, img_size=img_size or encoder.img_size)

    def _freeze(self) -> None:
        for p in self.image_encoder.parameters():
            p.requires_grad_(False)
        self.image_encoder.eval()

    def train(self, mode: bool = True) -> "MultiScaleMobileSAMBackbone":
        super().train(False)
        self.image_encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """图像 → 多尺度特征 | Image → multi-scale features (any square input size)."""
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"expected [B, 3, H, W], got {tuple(image.shape)}")
        return self.image_encoder.forward_multi_scale(image)
