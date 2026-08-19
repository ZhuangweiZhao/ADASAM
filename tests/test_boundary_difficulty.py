import torch

from adasam.metrics.boundary_difficulty import (
    HistogramBinaryMetrics,
    boundary_band,
    semantic_boundary,
)


def test_semantic_boundary_and_band() -> None:
    mask = torch.tensor([[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]])
    edge = semantic_boundary(mask)
    assert edge[0, :, 1:3].all()
    assert not edge[0, :, 0].any()
    assert boundary_band(mask, 1).sum() > edge.sum()


def test_histogram_metrics_rank_perfect_signal() -> None:
    metric = HistogramBinaryMetrics(128)
    score = torch.tensor([0.01, 0.1, 0.9, 0.99])
    target = torch.tensor([False, False, True, True])
    metric.update(score, target, torch.ones_like(target))
    result = metric.compute()
    assert result["pr_auc"] == 1.0
    assert result["roc_auc"] == 1.0
    assert result["top_fraction"]["0.10"]["enrichment"] == 2.0


def test_histogram_metrics_handles_single_class() -> None:
    metric = HistogramBinaryMetrics()
    metric.update(torch.ones(3), torch.ones(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool))
    result = metric.compute()
    assert result["pr_auc"] is None
    assert result["positives"] == 3
