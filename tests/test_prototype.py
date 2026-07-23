"""
Support Encoder 单元测试 | Unit tests for SupportEncoder.
=========================================================

覆盖 | Covers:
    - Stage 1 (MVP, n_encoder_layers=0): extract → concat → [K×N_s, C]
    - Stage 2 (n_encoder_layers=2): extract → self-attn → memory bank → [M, C]
    - 空 FG 掩码 → mask_token 填充
    - 多 support 输入 (variable K)
    - 配置往返 | config round-trip
纯合成张量, 无需真实数据/权重 | Pure synthetic tensors, no real data/weights needed.
"""

from __future__ import annotations

import pytest
import torch

from adasam.support_encoder import SupportEncoder, SupportEncoderConfig


# ═══════════════════════════════════════════════════════════════════
# SupportEncoder
# ═══════════════════════════════════════════════════════════════════

class TestSupportEncoderStage1:
    """Stage 1 MVP: no self-attention, no memory bank → direct concat."""

    @pytest.fixture(scope="class")
    def encoder(self):
        return SupportEncoder(SupportEncoderConfig(
            embed_dim=256, n_support_tokens=16, n_encoder_layers=0,
        ))

    def test_stage1_output_shape(self, encoder):
        """K=3, N_s=16 → [48, 256]."""
        feats = torch.randn(3, 256, 64, 64)
        masks = torch.rand(3, 64, 64) > 0.3
        out = encoder(feats, masks.float())
        assert out.shape == (3 * 16, 256)

    def test_stage1_single_support(self, encoder):
        """K=1 → [16, 256]."""
        feats = torch.randn(1, 256, 64, 64)
        masks = torch.rand(1, 64, 64) > 0.5
        out = encoder(feats, masks.float())
        assert out.shape == (16, 256)

    def test_stage1_all_empty_mask(self, encoder):
        """All-empty FG masks → all mask_tokens (valid=False)."""
        feats = torch.randn(2, 256, 64, 64)
        masks = torch.zeros(2, 64, 64)
        out = encoder(feats, masks)
        assert out.shape == (2 * 16, 256)
        # Should still give result (no crash)

    def test_stage1_small_fg(self, encoder):
        """Tiny FG region (< N_s pixels) → partial valid + pad."""
        feats = torch.randn(1, 256, 64, 64)
        masks = torch.zeros(1, 64, 64)
        masks[0, 10, 10] = 1.0  # single pixel FG
        out = encoder(feats, masks)
        assert out.shape == (16, 256)


class TestSupportEncoderStage2:
    """Stage 2: self-attention + memory bank."""

    @pytest.fixture(scope="class")
    def encoder(self):
        return SupportEncoder(SupportEncoderConfig(
            embed_dim=256, n_support_tokens=16, n_memory_tokens=64,
            n_encoder_layers=2, n_heads=8, ffn_dim=512, dropout=0.0,
        ))

    def test_stage2_output_shape(self, encoder):
        """K=3 → [64, 256] fixed regardless of K."""
        feats = torch.randn(3, 256, 64, 64)
        masks = torch.rand(3, 64, 64) > 0.3
        out = encoder(feats, masks.float())
        assert out.shape == (64, 256)

    def test_stage2_variable_k(self, encoder):
        """Memory bank output is always [M, C] for varying K."""
        for k in (1, 3, 7):
            feats = torch.randn(k, 256, 64, 64)
            masks = torch.rand(k, 64, 64) > 0.3
            out = encoder(feats, masks.float())
            assert out.shape == (64, 256), f"failed for K={k}"

    def test_stage2_differentiable(self, encoder):
        """Gradients flow through both self-attn and memory bank."""
        feats = torch.randn(3, 256, 64, 64)
        masks = torch.rand(3, 64, 64) > 0.3
        out = encoder(feats, masks.float())
        loss = out.sum()
        loss.backward()
        # At least memory_tokens should get gradient
        assert encoder.memory_bank.memory_tokens.grad is not None
        assert encoder.memory_bank.memory_tokens.grad.abs().sum() > 0


class TestSupportEncoderConfig:
    """Configuration validation."""

    def test_config_roundtrip(self):
        cfg = SupportEncoderConfig.from_dict({
            "embed_dim": 128, "n_support_tokens": 32,
            "n_memory_tokens": 32, "n_encoder_layers": 4,
        })
        assert cfg.embed_dim == 128
        assert cfg.n_support_tokens == 32
        assert cfg.n_memory_tokens == 32
        assert cfg.n_encoder_layers == 4
        assert cfg.is_stage2

    def test_config_defaults(self):
        cfg = SupportEncoderConfig()
        assert cfg.embed_dim == 256
        assert cfg.n_support_tokens == 16
        assert cfg.is_stage2 is False  # default: n_encoder_layers=0

    def test_config_unknown_keys_ignored(self):
        cfg = SupportEncoderConfig.from_dict({"embed_dim": 512, "legacy_field": 99})
        assert cfg.embed_dim == 512

    def test_config_is_stage2_false(self):
        cfg = SupportEncoderConfig(n_encoder_layers=0)
        assert not cfg.is_stage2


class TestSupportEncoderShapeErrors:
    """Input validation."""

    @pytest.fixture(scope="class")
    def encoder(self):
        return SupportEncoder(SupportEncoderConfig())

    def test_wrong_features_ndim_raises(self, encoder):
        """features must be [K,C,gh,gw]."""
        with pytest.raises(ValueError):
            encoder(torch.randn(256, 64, 64), torch.randn(64, 64))

    def test_wrong_masks_ndim_raises(self, encoder):
        """masks must be [K,gh,gw]."""
        with pytest.raises(ValueError):
            encoder(torch.randn(2, 256, 64, 64), torch.randn(2, 1, 64, 64))
