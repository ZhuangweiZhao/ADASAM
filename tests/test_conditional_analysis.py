from __future__ import annotations

import torch

from adasam.metrics import component_iou_sums, component_size_map, summarize_component_iou


def test_component_size_map_assigns_three_relative_bands() -> None:
    target = torch.zeros(1, 100, 100, dtype=torch.long)
    target[0, 1:3, 1:3] = 1       # 4 <= 10 pixels: small
    target[0, 10:15, 10:15] = 1   # 25 <= 100 pixels: medium
    target[0, 30:50, 30:50] = 1   # 400 pixels: large
    strata = component_size_map(target, num_classes=2)
    assert set(strata.unique().tolist()) == {-1, 0, 1, 2}
    assert int((strata == 0).sum()) == 4
    assert int((strata == 1).sum()) == 25
    assert int((strata == 2).sum()) == 400


def test_component_iou_is_one_for_perfect_prediction() -> None:
    target = torch.zeros(1, 100, 100, dtype=torch.long)
    target[0, 1:3, 1:3] = 1
    target[0, 10:15, 10:15] = 1
    target[0, 30:50, 30:50] = 1
    sums, counts = component_iou_sums(target, target, num_classes=2)
    result = summarize_component_iou(sums, counts)
    assert result["small"] == 1.0
    assert result["medium"] == 1.0
    assert result["large"] == 1.0
    assert result["component_counts"] == {"small": 1, "medium": 1, "large": 1}
