"""
匈牙利匹配器与成对代价测试 | Hungarian matcher & pairwise cost tests.
=====================================================================

纯合成张量, 不依赖权重与数据 | Synthetic tensors only; no weights/data required.
"""

from __future__ import annotations

import pytest
import torch

from adasam.losses import (
    HungarianMatcher,
    MatcherConfig,
    dice_loss,
    pairwise_dice_cost,
    pairwise_sigmoid_bce_cost,
)

torch.manual_seed(0)


# ── 成对代价函数 | Pairwise cost functions ──


def test_pairwise_dice_cost_matches_looped_dice_loss():
    """成对 dice 代价与逐对 dice_loss 一致 | pairwise dice agrees with looped dice_loss."""
    n, m, hw = 4, 3, 64
    logits = torch.randn(n, hw)
    targets = (torch.rand(m, hw) > 0.5).float()

    cost = pairwise_dice_cost(logits, targets)
    assert cost.shape == (n, m)

    for i in range(n):
        for j in range(m):
            ref = dice_loss(logits[i : i + 1], targets[j : j + 1], reduction="none")[0]
            assert torch.allclose(cost[i, j], ref, atol=1e-5)


def test_pairwise_bce_cost_matches_elementwise():
    """成对 BCE 代价与逐对逐像素 BCE 均值一致 | pairwise BCE agrees with per-pair mean BCE."""
    n, m, hw = 3, 4, 128
    logits = torch.randn(n, hw)
    targets = (torch.rand(m, hw) > 0.5).float()

    cost = pairwise_sigmoid_bce_cost(logits, targets)
    assert cost.shape == (n, m)

    for i in range(n):
        for j in range(m):
            ref = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[i], targets[j], reduction="mean"
            )
            assert torch.allclose(cost[i, j], ref, atol=1e-5)


def test_pairwise_costs_accept_soft_targets():
    """软目标 (双线性降采样产物) 不报错且有限 | soft targets are accepted and finite."""
    logits = torch.randn(5, 256)
    soft = torch.rand(2, 256)  # ∈ (0,1) 软掩码 | soft masks
    for fn in (pairwise_dice_cost, pairwise_sigmoid_bce_cost):
        out = fn(logits, soft)
        assert out.shape == (5, 2)
        assert torch.isfinite(out).all()


def test_pairwise_dice_perfect_prediction_low_cost():
    """完美预测的对角代价接近 0 | perfect predictions give near-zero diagonal cost."""
    targets = (torch.rand(3, 100) > 0.7).float()
    logits = targets * 20.0 - 10.0  # ±10 logits
    cost = pairwise_dice_cost(logits, targets)
    diag = cost.diag()
    assert (diag < 0.01).all()
    # 非对角 (不同掩码) 代价显著更高 | off-diagonal costs are clearly higher
    off = cost + torch.eye(3) * 10.0
    assert (diag < off.min(dim=1).values + 1e-6).all()


# ── 匈牙利匹配器 | Hungarian matcher ──


def _distinct_masks(m: int, h: int = 16, w: int = 16) -> torch.Tensor:
    """m 个互不重叠的方块掩码 | m disjoint square masks."""
    masks = torch.zeros(m, h, w)
    for i in range(m):
        masks[i, i * 4 : i * 4 + 4, i * 4 : i * 4 + 4] = 1.0
    return masks


def test_perfect_predictions_identity_permutation():
    """完美预测 → 恒等排列 | perfect predictions → identity permutation."""
    gt = _distinct_masks(3)
    logits = gt * 20.0 - 10.0
    obj = torch.full((3,), 5.0)
    pred_idx, gt_idx = HungarianMatcher(MatcherConfig()).match(logits, obj, gt)
    order = pred_idx[gt_idx.argsort()]
    assert torch.equal(order, torch.arange(3))


def test_more_queries_than_gt():
    """N=8 > M=3 → 恰好 3 对 | N=8 > M=3 → exactly 3 pairs."""
    gt = _distinct_masks(3)
    logits = torch.randn(8, 16, 16)
    logits[:3] = gt * 20.0 - 10.0
    obj = torch.randn(8)
    pred_idx, gt_idx = HungarianMatcher(MatcherConfig()).match(logits, obj, gt)
    assert pred_idx.shape == (3,) and gt_idx.shape == (3,)
    assert len(set(pred_idx.tolist())) == 3      # 一对一 | one-to-one
    assert sorted(gt_idx.tolist()) == [0, 1, 2]


def test_objectness_breaks_ties():
    """两个相同掩码, objectness 高者胜出 | identical masks → higher objectness wins."""
    gt = _distinct_masks(1)
    logits = torch.stack([gt[0], gt[0]]) * 20.0 - 10.0    # 两个一样的预测 | identical preds
    obj = torch.tensor([-2.0, 3.0])
    pred_idx, _ = HungarianMatcher(MatcherConfig()).match(logits, obj, gt)
    assert pred_idx.item() == 1


def test_more_gt_than_queries_raises():
    """M > N → ValueError | more GT than queries raises."""
    gt = _distinct_masks(4)
    logits = torch.randn(2, 16, 16)
    obj = torch.randn(2)
    with pytest.raises(ValueError):
        HungarianMatcher(MatcherConfig()).match(logits, obj, gt)


def test_nan_costs_are_clamped():
    """极端 logits 产生的非有限代价被钳制, 匹配仍可解 | non-finite costs are clamped."""
    gt = _distinct_masks(2)
    logits = torch.full((4, 16, 16), 1e10)                # 极端 | extreme
    obj = torch.tensor([float("inf"), -float("inf"), 0.0, 1.0])
    pred_idx, gt_idx = HungarianMatcher(MatcherConfig()).match(logits, obj, gt)
    assert pred_idx.shape == (2,) and gt_idx.shape == (2,)


def test_matcher_config_from_dict_with_prefix():
    """yaml loss 段 match_cost_* 前缀键正确映射 | match_cost_* prefixed keys map correctly."""
    cfg = MatcherConfig.from_dict(
        {"match_cost_objectness": 1.5, "match_cost_mask": 4.0, "match_cost_dice": 3.0,
         "focal_weight": 1.0}
    )
    assert cfg.cost_objectness == 1.5
    assert cfg.cost_mask == 4.0
    assert cfg.cost_dice == 3.0
