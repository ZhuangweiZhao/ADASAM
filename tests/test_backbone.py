"""
MobileSAM 骨干与预处理单元测试 | Unit tests for MobileSAM backbone & preprocessing.
====================================================================================

覆盖 | Covers:
    - transforms.preprocess_image: 形状、归一化、resize/pad 元数据 | shape, normalize, meta.
    - transforms.resize_mask: 最近邻缩放、二值保持 | nearest resize, binary preserved.
    - MobileSAMBackbone: 嵌入形状 [B,256,64,64]、参数冻结、train() 守卫、no_grad。

需要权重的测试在 weights/mobile_sam.pt 缺失时跳过 (skipif)。
Weight-dependent tests are skipped (skipif) when weights/mobile_sam.pt is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from adasam.utils.transforms import preprocess_image, resize_mask, SAM_IMAGE_SIZE

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CKPT = _REPO_ROOT / "weights" / "mobile_sam.pt"
_has_ckpt = _CKPT.exists()
_skip_ckpt = pytest.mark.skipif(not _has_ckpt, reason="weights/mobile_sam.pt not present")


# ═══════════════════════════════════════════════════════════════════
# transforms
# ═══════════════════════════════════════════════════════════════════

def test_preprocess_square_tile_shape_and_meta():
    """896² 方形 tile → [3,1024,1024], 无 padding | square tile → 1024², no pad."""
    img = np.random.randint(0, 256, size=(896, 896, 3), dtype=np.uint8)
    x, meta = preprocess_image(img)
    assert x.shape == (3, SAM_IMAGE_SIZE, SAM_IMAGE_SIZE)
    assert x.dtype == torch.float32
    assert meta.original_size == (896, 896)
    assert meta.input_size == (1024, 1024)          # 方形整体缩放, 无 pad | square, no pad


def test_preprocess_normalization_range():
    """归一化后应大致零均值、量级 O(1) | normalized ≈ zero-mean, O(1) magnitude."""
    img = np.full((896, 896, 3), 128, dtype=np.uint8)
    x, _ = preprocess_image(img)
    # 128 落在 SAM mean(~110-124) 附近, 归一化后 |value| 应较小 | near SAM mean → small
    assert x.abs().mean() < 1.0


def test_preprocess_non_square_pads_to_square():
    """非方形图像最长边→1024, 短边补零 | longest side→1024, short side padded."""
    img = np.random.randint(0, 256, size=(500, 1000, 3), dtype=np.uint8)
    x, meta = preprocess_image(img)
    assert x.shape == (3, 1024, 1024)
    assert meta.input_size == (512, 1024)           # 1000→1024, 500→512
    # 底部 padding 区域应为 0 (归一化前补零) | bottom pad region is exactly zero
    assert torch.count_nonzero(x[:, 512:, :]) == 0


def test_resize_mask_nearest_binary():
    """掩码缩放保持二值 | resized mask stays binary {0,1}."""
    m = np.zeros((896, 896), dtype=bool)
    m[100:400, 100:400] = True
    out = resize_mask(m, 64)
    assert out.shape == (64, 64)
    assert set(torch.unique(out).tolist()).issubset({0.0, 1.0})
    assert out.sum() > 0                            # 前景保留 | foreground preserved


# ═══════════════════════════════════════════════════════════════════
# MobileSAMBackbone (weight-dependent)
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def backbone():
    from adasam.backbone import MobileSAMBackbone
    return MobileSAMBackbone.build(_CKPT, device="cpu")


@_skip_ckpt
def test_backbone_forward_embedding_shape(backbone):
    """forward → {"image_embedding": [B,256,64,64]} | encoder output contract."""
    x = torch.randn(1, 3, 1024, 1024)
    out = backbone(x)
    assert set(out.keys()) == {"image_embedding"}
    assert out["image_embedding"].shape == (1, 256, 64, 64)


@_skip_ckpt
def test_backbone_is_frozen(backbone):
    """所有骨干参数 requires_grad=False | all backbone params frozen."""
    assert all(not p.requires_grad for p in backbone.parameters())


@_skip_ckpt
def test_backbone_train_guard_keeps_eval(backbone):
    """调用 train() 后仍为 eval (冻结骨干守卫) | train() keeps eval (frozen guard)."""
    backbone.train()
    assert backbone.training is False
    assert backbone.image_encoder.training is False


@_skip_ckpt
def test_backbone_forward_no_grad(backbone):
    """嵌入不带梯度 (no_grad 前向) | embedding carries no grad."""
    x = torch.randn(1, 3, 1024, 1024)
    out = backbone(x)
    assert out["image_embedding"].requires_grad is False


@_skip_ckpt
def test_backbone_rejects_bad_shape(backbone):
    """非法输入形状应报错 | bad input shape raises."""
    with pytest.raises(ValueError):
        backbone(torch.randn(3, 1024, 1024))        # 缺 batch 维 | missing batch dim
