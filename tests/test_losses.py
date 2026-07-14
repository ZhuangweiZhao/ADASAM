"""
分割损失单元测试 | Unit tests for segmentation losses.
======================================================

纯合成张量, 快速 | Pure synthetic tensors, fast. 覆盖 focal / dice / combined / mask_iou。
"""

from __future__ import annotations

import pytest
import torch

from adasam.losses import focal_loss, dice_loss, combined_loss, mask_iou


def test_focal_low_for_correct_high_for_wrong():
    """正确预测 focal 低, 错误预测 focal 高 | focal low when correct, high when wrong."""
    gt = torch.zeros(1, 8, 8)
    gt[0, 2:6, 2:6] = 1.0
    correct = torch.where(gt > 0.5, 10.0, -10.0)          # 置信且正确 | confident & correct
    wrong = torch.where(gt > 0.5, -10.0, 10.0)            # 置信且错误 | confident & wrong
    assert focal_loss(correct, gt) < focal_loss(wrong, gt)
    assert focal_loss(correct, gt).item() < 1e-2


def test_focal_eps_clamp_finite():
    """极端 logits 下 focal 仍有限 (eps 截断) | focal stays finite under extreme logits."""
    gt = torch.ones(1, 4, 4)
    loss = focal_loss(torch.full((1, 4, 4), -1e4), gt)     # 完全错误 | fully wrong
    assert torch.isfinite(loss)


def test_dice_perfect_and_disjoint():
    """完美掩码 dice≈0, 不相交 dice≈1 | perfect ≈ 0, disjoint ≈ 1."""
    gt = torch.zeros(1, 16, 16)
    gt[0, :8, :8] = 1.0
    perfect = torch.where(gt > 0.5, 20.0, -20.0)
    assert dice_loss(perfect, gt).item() < 1e-2
    disjoint_gt = torch.zeros(1, 16, 16)
    disjoint_gt[0, 8:, 8:] = 1.0
    assert dice_loss(perfect, disjoint_gt).item() > 0.9


def test_dice_per_instance_reduction():
    """[N,H,W] 逐实例 dice + none 归约 | per-instance dice with 'none' reduction."""
    logits = torch.randn(3, 8, 8)
    gt = (torch.rand(3, 8, 8) > 0.5).float()
    per = dice_loss(logits, gt, reduction="none")
    assert per.shape == (3,)


def test_combined_is_focal_plus_dice():
    """combined == focal + dice (默认权重) | combined equals focal + dice at default weights."""
    logits = torch.randn(2, 8, 8)
    gt = (torch.rand(2, 8, 8) > 0.5).float()
    expected = focal_loss(logits, gt) + dice_loss(logits, gt)
    assert combined_loss(logits, gt).item() == pytest.approx(expected.item(), abs=1e-5)


def test_mask_iou_known_values():
    """IoU 已知值: 相同=1, 不相交=0, 半重叠 | known IoU values."""
    a = torch.zeros(1, 10, 10); a[0, :5, :] = 1
    b = torch.zeros(1, 10, 10); b[0, :5, :] = 1
    c = torch.zeros(1, 10, 10); c[0, 5:, :] = 1
    assert mask_iou(a, b).item() == pytest.approx(1.0)
    assert mask_iou(a, c).item() == pytest.approx(0.0)
    d = torch.zeros(1, 10, 10); d[0, 3:8, :] = 1          # 与 a 交 2 行, 并 8 行 → 0.25
    assert mask_iou(a, d).item() == pytest.approx(0.25, abs=1e-5)
