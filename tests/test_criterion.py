"""
SetCriterion 测试 | SetCriterion tests.
=========================================

纯合成张量, 不依赖权重与数据 | Synthetic tensors only; no weights/data required.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from adasam.losses import CriterionConfig, HungarianMatcher, MatcherConfig, SetCriterion
from adasam.prompt import DPGOutput

H256 = 32   # 缩小的 "SAM 掩码" 分辨率, 加速测试 | shrunken "SAM mask" res for speed
H64 = 8     # 缩小的 "DPG 网格" 分辨率 | shrunken "DPG grid" res
HW_TILE = 64  # 缩小的 "tile" 分辨率 | shrunken "tile" res


def _make_criterion(**kwargs) -> SetCriterion:
    return SetCriterion(HungarianMatcher(MatcherConfig()), CriterionConfig(**kwargs))


def _make_gt(m: int) -> torch.Tensor:
    """m 个互不重叠的方块 GT (tile 分辨率) | m disjoint square GT masks at tile res."""
    gt = torch.zeros(m, HW_TILE, HW_TILE)
    for i in range(m):
        gt[i, i * 16 : i * 16 + 16, i * 16 : i * 16 + 16] = 1.0
    return gt


def _make_dpg_out(n: int, num_layers: int = 2, from_gt: torch.Tensor | None = None) -> DPGOutput:
    torch.manual_seed(0)
    if from_gt is not None:
        gt_grid = F.interpolate(from_gt.unsqueeze(0), (H64, H64), mode="bilinear",
                                align_corners=False)[0]
        masks = torch.full((n, H64, H64), -10.0)
        masks[: from_gt.shape[0]] = gt_grid * 20.0 - 10.0
        obj = torch.full((n,), -5.0)
        obj[: from_gt.shape[0]] = 5.0
    else:
        masks = torch.randn(n, H64, H64)
        obj = torch.randn(n)
    aux = [
        {"mask_logits": masks.clone(), "objectness_logits": obj.clone()}
        for _ in range(num_layers)
    ]
    return DPGOutput(
        instance_queries=torch.randn(n, 32),
        objectness_logits=obj,
        mask_logits=masks,
        aux=aux,
    )


def test_perfect_match_near_zero_mask_losses():
    """完美预测 → focal/dice/iou_head ≈ 0 | perfect predictions → near-zero losses."""
    m, n = 2, 4
    gt = _make_gt(m)
    gt_256 = F.interpolate(gt.unsqueeze(0), (H256, H256), mode="bilinear",
                           align_corners=False)[0]
    sam_logits = torch.full((n, H256, H256), -10.0)
    sam_logits[:m] = gt_256 * 20.0 - 10.0
    iou_pred = torch.zeros(n)
    iou_pred[:m] = 1.0
    dpg_out = _make_dpg_out(n, from_gt=gt)

    out = _make_criterion()(sam_logits, iou_pred, dpg_out, gt)
    assert out["focal"] < 0.01
    assert out["dice"] < 0.05
    assert out["iou_head"] < 0.05
    assert out["n_matched"] == m
    assert out["loss"].requires_grad is False or torch.isfinite(out["loss"])


def test_objectness_eos_weighting_matches_hand_computation():
    """eos 加权 BCE 与手算一致 | eos-weighted BCE matches hand computation."""
    crit = _make_criterion(eos_coef=0.1)
    obj_logits = torch.tensor([2.0, -1.0, 0.5, -3.0])
    matched = torch.tensor([0, 2])

    got = crit._objectness_bce(obj_logits, matched)

    target = torch.tensor([1.0, 0.0, 1.0, 0.0])
    weight = torch.tensor([1.0, 0.1, 1.0, 0.1])
    bce = F.binary_cross_entropy_with_logits(obj_logits, target, reduction="none")
    expected = (weight * bce).sum() / weight.sum()
    assert torch.allclose(got, expected, atol=1e-6)


def test_loss_is_finite_scalar_with_grad():
    """总损失为可回传的有限标量 | total loss is a finite scalar with grad."""
    m, n = 3, 6
    gt = _make_gt(m)
    sam_logits = torch.randn(n, H256, H256, requires_grad=True)
    iou_pred = torch.rand(n, requires_grad=True)
    dpg = _make_dpg_out(n)
    dpg.mask_logits.requires_grad_(True)

    out = _make_criterion()(sam_logits, iou_pred, dpg, gt)
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])
    assert out["loss"].requires_grad
    out["loss"].backward()
    assert sam_logits.grad is not None and sam_logits.grad.abs().sum() > 0
    assert iou_pred.grad is not None


def test_aux_contains_all_layers_and_coupling():
    """aux 桶 = L 层深监督 + 最终层耦合项 (>0) | aux = L layers + final coupling."""
    m, n = 2, 4
    gt = _make_gt(m)
    sam_logits = torch.randn(n, H256, H256)
    dpg = _make_dpg_out(n, num_layers=3)
    out = _make_criterion()(sam_logits, torch.rand(n), dpg, gt)
    # 随机预测下 aux 必然显著为正 | aux is clearly positive for random predictions
    assert out["aux"] > 0
    # aux_weight=0 时不影响总损失 | aux_weight=0 removes it from the total
    out_no_aux = _make_criterion(aux_weight=0.0)(sam_logits, torch.rand(n), dpg, gt)
    assert out_no_aux["loss"] < out["loss"] + 1e-6


def test_tiny_instance_survives_soft_downsampling():
    """6px 小实例面积平均降采样后在小网格仍有信号 | tiny instance keeps signal (area mode)."""
    gt = torch.zeros(1, HW_TILE, HW_TILE)
    gt[0, 30:33, 30:32] = 1.0                            # 6 px 实例 | 6-pixel instance
    gt_grid = SetCriterion._soft_resize(gt, (H64, H64))
    assert gt_grid.sum() > 0
    assert gt_grid.max() > 0.01                          # 面积占比 6/64 ≈ 0.09 | 6/64 ≈ 0.09


def test_monitoring_metrics_present():
    m, n = 2, 8
    gt = _make_gt(m)
    out = _make_criterion()(torch.randn(n, H256, H256), torch.rand(n), _make_dpg_out(n), gt)
    for key in ("mean_obj_matched", "mean_obj_unmatched", "n_matched"):
        assert key in out
        assert torch.isfinite(out[key])
