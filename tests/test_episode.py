"""
Episode 采样单元测试 | Unit tests for EpisodeSampler.
=====================================================

用一个受控的假数据集验证采样器逻辑 (不依赖真实数据, 快速确定性):
Validates sampler logic with a controlled fake dataset (no real data, fast & deterministic):

    - min_tiles 过滤 | min_tiles filtering.
    - 需 ≥2 场景 (单场景类被排除) | ≥2 scenes required (single-scene class excluded).
    - 场景不相交: query 场景 ∉ support 场景 | scene-disjoint support/query.
    - K-shot 数量 | K-shot count.
    - 种子可复现 | reproducible under a fixed seed.
"""

from __future__ import annotations

import pytest

from adasam.datasets import EpisodeSampler


class _FakeDataset:
    """受控假数据集 | Controlled fake dataset.

    spec: {class_id: [(tile_idx, scene_id), ...]}
    """

    def __init__(self, spec: dict[int, list[tuple[int, int]]]):
        self._spec = spec
        self._scene_of: dict[int, int] = {}
        for tiles in spec.values():
            for idx, scene in tiles:
                self._scene_of[idx] = scene

    def visible_classes(self) -> list[int]:
        return sorted(self._spec)

    def class_to_tiles(self, class_id: int) -> list[int]:
        return [idx for idx, _ in self._spec.get(class_id, [])]

    def source_image_id(self, idx: int) -> int:
        return self._scene_of[idx]


def _make_spec():
    # A: 40 tiles over 4 scenes → eligible (≥30 tiles, ≥2 scenes)
    a = [(i, i // 10) for i in range(40)]           # scenes 0..3
    # B: 10 tiles → excluded by min_tiles=30
    b = [(100 + i, 10) for i in range(10)]
    # C: 40 tiles all in ONE scene → excluded (needs ≥2 scenes)
    c = [(200 + i, 20) for i in range(40)]
    return {1: a, 2: b, 3: c}


def test_eligible_classes_after_filtering():
    """只有 A(class 1) 通过 min_tiles + ≥2 场景 | only class 1 passes both filters."""
    ds = _FakeDataset(_make_spec())
    sampler = EpisodeSampler(ds, k_shot=5, seed=42, min_tiles=30)
    assert sampler.eligible_classes() == [1]


def test_episode_structure_and_kshot():
    """episode 结构与 K-shot 数量 | episode structure and K-shot count."""
    ds = _FakeDataset(_make_spec())
    sampler = EpisodeSampler(ds, k_shot=5, seed=42, min_tiles=30)
    ep = sampler.sample()
    assert set(ep.keys()) == {"class_id", "support_indices", "query_index"}
    assert ep["class_id"] == 1
    assert len(ep["support_indices"]) == 5
    assert ep["query_index"] not in ep["support_indices"]


def test_scene_disjoint():
    """support 场景与 query 场景不相交 | support scenes disjoint from query scene."""
    ds = _FakeDataset(_make_spec())
    sampler = EpisodeSampler(ds, k_shot=8, seed=7, min_tiles=30)
    for _ in range(50):
        ep = sampler.sample()
        q_scene = ds.source_image_id(ep["query_index"])
        s_scenes = {ds.source_image_id(i) for i in ep["support_indices"]}
        assert q_scene not in s_scenes


def test_kshot_capped_by_pool():
    """support 池不足时 K 自动降级 | K degrades when the support pool is small."""
    # A across 2 scenes of 5 tiles each (10 tiles < 30 → lower min_tiles for this test)
    ds = _FakeDataset({1: [(i, i // 5) for i in range(10)]})
    sampler = EpisodeSampler(ds, k_shot=20, seed=1, min_tiles=5)
    ep = sampler.sample()
    # query 场景占 5 张, 另一场景仅剩 5 张可作 support | the other scene has only 5
    assert len(ep["support_indices"]) == 5


def test_reproducible_with_seed():
    """同种子 → 相同 episode 序列 | same seed → identical episode sequence."""
    ds = _FakeDataset(_make_spec())
    s1 = EpisodeSampler(ds, k_shot=5, seed=123, min_tiles=30)
    s2 = EpisodeSampler(ds, k_shot=5, seed=123, min_tiles=30)
    seq1 = [s1.sample() for _ in range(10)]
    seq2 = [s2.sample() for _ in range(10)]
    assert seq1 == seq2


def test_raises_when_no_eligible_class():
    """无合格类 → 抛错 | raises when nothing passes filtering."""
    ds = _FakeDataset({1: [(i, 0) for i in range(40)]})   # single scene only
    with pytest.raises(ValueError):
        EpisodeSampler(ds, k_shot=5, seed=42, min_tiles=30)
