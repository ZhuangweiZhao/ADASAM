"""
instance_match 单元测试 | Unit tests for instance-level matching & Instance mIoU.
==============================================================================

覆盖 | Covers:
    - pairwise_iou: 形状、取值范围、已知 IoU | shape, value range, known IoU.
    - greedy_match: 完美匹配、一对一唯一性、阈值边界、空输入 | perfect, uniqueness, threshold, empty.
    - instance_miou: 需求 #5 定义、空 GT/空预测 | requirement #5 definition, empty cases.
"""

import numpy as np
import pytest

from adasam.metrics.instance_match import pairwise_iou, greedy_match, instance_miou


# ═══════════════════════════════════════════════════════════════════
# 工具: 造一个在 [y0:y1, x0:x1] 为 True 的方块掩码 | Helper: rectangle mask
# ═══════════════════════════════════════════════════════════════════

def _rect(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


# ═══════════════════════════════════════════════════════════════════
# pairwise_iou
# ═══════════════════════════════════════════════════════════════════

def test_pairwise_iou_shape_and_range():
    """形状 [P,G], 取值 ∈ [0,1] | shape [P,G], values in [0,1]."""
    preds = [_rect(10, 10, 0, 5, 0, 5), _rect(10, 10, 5, 10, 5, 10)]
    gts = [_rect(10, 10, 0, 5, 0, 5)]
    iou = pairwise_iou(preds, gts)
    assert iou.shape == (2, 1)
    assert np.all(iou >= 0.0) and np.all(iou <= 1.0)


def test_pairwise_iou_identity_and_disjoint():
    """相同掩码 IoU=1, 不相交 IoU=0 | identical → 1, disjoint → 0."""
    a = _rect(10, 10, 0, 5, 0, 5)
    b = _rect(10, 10, 5, 10, 5, 10)
    iou = pairwise_iou([a, b], [a])
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[1, 0] == pytest.approx(0.0)


def test_pairwise_iou_known_value():
    """半重叠已知 IoU | half-overlap known IoU.

    A = 行 0..4 (5行), B = 行 3..7 (5行). 交=2行, 并=8行 → IoU=2/8=0.25.
    """
    a = _rect(10, 10, 0, 5, 0, 10)
    b = _rect(10, 10, 3, 8, 0, 10)
    iou = pairwise_iou([a], [b])
    assert iou[0, 0] == pytest.approx(0.25, abs=1e-6)


def test_pairwise_iou_empty():
    """空输入返回相应空形状 | empty inputs → empty shapes."""
    a = _rect(4, 4, 0, 2, 0, 2)
    assert pairwise_iou([], [a]).shape == (0, 1)
    assert pairwise_iou([a], []).shape == (1, 0)
    assert pairwise_iou([], []).shape == (0, 0)


# ═══════════════════════════════════════════════════════════════════
# greedy_match
# ═══════════════════════════════════════════════════════════════════

def test_greedy_perfect_match():
    """两预测精确命中两 GT → tp=2, fp=0, fn=0 | perfect → tp=2."""
    g1 = _rect(20, 20, 0, 5, 0, 5)
    g2 = _rect(20, 20, 10, 15, 10, 15)
    res = greedy_match([g1, g2], [0.9, 0.8], [g1, g2], iou_thr=0.5)
    assert res["tp"] == 2 and res["fp"] == 0 and res["fn"] == 0
    assert res["n_gt"] == 2 and res["n_pred"] == 2


def test_greedy_one_pred_one_gt_uniqueness():
    """一个预测覆盖两个 GT 时, 只能匹配其一 → 1 TP + 1 FN (需求 #3).
    A single prediction overlapping two GT can match only one → 1 TP + 1 FN."""
    g1 = _rect(20, 20, 0, 10, 0, 5)
    g2 = _rect(20, 20, 0, 10, 5, 10)
    big = _rect(20, 20, 0, 10, 0, 10)   # 同时覆盖 g1,g2 | overlaps both
    res = greedy_match([big], [0.9], [g1, g2], iou_thr=0.3)
    assert res["n_pred"] == 1 and res["n_gt"] == 2
    assert res["tp"] == 1          # 只认领一个 GT | claims exactly one GT
    assert res["fn"] == 1          # 另一个 GT 漏检 | the other GT is missed
    assert res["fp"] == 0
    # 确认成对结果唯一 | matched pair is unique
    assert len(res["matched_pairs"]) == 1


def test_greedy_two_preds_compete_higher_score_wins():
    """两预测竞争同一 GT, 高分者匹配, 低分者成 FP.
    Two predictions competing for one GT: higher score wins, other → FP."""
    g = _rect(20, 20, 0, 10, 0, 10)
    p_hi = _rect(20, 20, 0, 10, 0, 10)     # 完全重合, 高分 | exact, high score
    p_lo = _rect(20, 20, 0, 10, 0, 9)      # 高重叠, 低分 | high overlap, low score
    res = greedy_match([p_lo, p_hi], [0.3, 0.9], [g], iou_thr=0.5)
    assert res["tp"] == 1 and res["fp"] == 1 and res["fn"] == 0
    # 高分预测 (index 1) 认领了 GT | the high-score pred (idx 1) claimed the GT
    assert res["matched_pairs"][0][0] == 1


def test_greedy_threshold_boundary():
    """IoU 略低于阈值 → 不匹配 (FP+FN) | IoU below threshold → no match."""
    g = _rect(10, 10, 0, 5, 0, 10)   # 5 行 | 5 rows
    p = _rect(10, 10, 3, 8, 0, 10)   # 与 g IoU=2/8=0.25 | IoU 0.25
    # 阈值 0.5 → 不匹配 | thr 0.5 → no match
    res = greedy_match([p], [0.9], [g], iou_thr=0.5)
    assert res["tp"] == 0 and res["fp"] == 1 and res["fn"] == 1
    # 阈值 0.2 → 匹配 | thr 0.2 → match
    res2 = greedy_match([p], [0.9], [g], iou_thr=0.2)
    assert res2["tp"] == 1 and res2["fp"] == 0 and res2["fn"] == 0


def test_greedy_empty_cases():
    """空预测 → 全 FN; 空 GT → 全 FP | empty pred → all FN; empty GT → all FP."""
    g = _rect(8, 8, 0, 4, 0, 4)
    r_no_pred = greedy_match([], [], [g], iou_thr=0.5)
    assert r_no_pred["tp"] == 0 and r_no_pred["fn"] == 1 and r_no_pred["fp"] == 0
    r_no_gt = greedy_match([g], [0.9], [], iou_thr=0.5)
    assert r_no_gt["tp"] == 0 and r_no_gt["fp"] == 1 and r_no_gt["fn"] == 0


def test_greedy_tp_plus_fn_equals_ngt():
    """不变式: TP + FN == n_gt | invariant TP + FN == n_gt."""
    gts = [_rect(30, 30, i * 5, i * 5 + 4, 0, 4) for i in range(4)]
    preds = [_rect(30, 30, 0, 4, 0, 4), _rect(30, 30, 20, 30, 20, 30)]
    res = greedy_match(preds, [0.9, 0.5], gts, iou_thr=0.5)
    assert res["tp"] + res["fn"] == res["n_gt"]
    assert res["tp"] + res["fp"] == res["n_pred"]


# ═══════════════════════════════════════════════════════════════════
# instance_miou
# ═══════════════════════════════════════════════════════════════════

def test_instance_miou_perfect():
    """完美预测 → mIoU=1 | perfect prediction → mIoU 1."""
    g1 = _rect(20, 20, 0, 5, 0, 5)
    g2 = _rect(20, 20, 10, 15, 10, 15)
    per_gt, mean = instance_miou([g1, g2], [g1, g2])
    assert mean == pytest.approx(1.0)
    assert len(per_gt) == 2


def test_instance_miou_per_gt_max_definition():
    """每个 GT 取最大 IoU 预测 (与匹配无关) | per-GT max IoU, matching-independent.

    两个 GT 都由同一个完美预测命中 → 都为 1 (定义允许共享预测).
    """
    g1 = _rect(20, 20, 0, 5, 0, 5)
    g2 = _rect(20, 20, 0, 5, 0, 5)          # 与 g1 相同 | same as g1
    per_gt, mean = instance_miou([g1], [g1, g2])
    assert per_gt == pytest.approx([1.0, 1.0])
    assert mean == pytest.approx(1.0)


def test_instance_miou_no_pred_is_zero():
    """有 GT 无预测 → 全 0 | GT but no prediction → all zeros."""
    g = _rect(10, 10, 0, 5, 0, 5)
    per_gt, mean = instance_miou([], [g])
    assert per_gt == [0.0] and mean == pytest.approx(0.0)


def test_instance_miou_no_gt_is_nan():
    """无 GT → NaN (调用方跳过) | no GT → NaN (caller skips)."""
    p = _rect(10, 10, 0, 5, 0, 5)
    per_gt, mean = instance_miou([p], [])
    assert per_gt == []
    assert np.isnan(mean)
