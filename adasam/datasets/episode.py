"""
小样本 Episode 采样 | Few-shot Episode Sampling.
=================================================

从数据集中按类采样一个 episode: K 张 support tile + 1 张 query tile。
Sample one episode from the dataset: K support tiles + 1 query tile, for a chosen class.

两条硬约束 | Two hard constraints:
    1. **场景不相交 | Scene-disjoint**: support 与 query 来自不同源全图 (orig_image_id),
       杜绝数据泄漏。support and query come from different source images (no leakage).
    2. **min_tiles 过滤 | min_tiles filter**: tile 数 < min_tiles 的类被排除, 避免稀有类不稳定
       (lesson #9)。classes with fewer than min_tiles tiles are excluded.

只依赖数据集的查询接口 (visible_classes / class_to_tiles / source_image_id), 与具体数据集实现解耦。
Depends only on the dataset query interface, decoupled from the concrete dataset.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Protocol, TypedDict


class SupportsEpisodeQuery(Protocol):
    """episode 采样所需的最小数据集接口 | Minimal dataset interface for episode sampling."""

    def visible_classes(self) -> list[int]: ...
    def class_to_tiles(self, class_id: int) -> list[int]: ...
    def source_image_id(self, idx: int) -> int: ...


class Episode(TypedDict):
    """一个 episode | One sampled episode."""

    class_id: int
    support_indices: list[int]
    query_index: int


class EpisodeSampler:
    """场景不相交的 K-shot episode 采样器 | Scene-disjoint K-shot episode sampler.

    :param dataset: 提供查询接口的数据集 | dataset providing the query interface.
    :param k_shot: 每个 episode 的 support tile 数上限 | max support tiles per episode.
    :param seed: 随机种子 (可复现) | RNG seed (reproducible).
    :param min_tiles: 类别最少 tile 数, 低于则排除 | min tiles per class, else excluded.
    """

    def __init__(
        self,
        dataset: SupportsEpisodeQuery,
        k_shot: int = 5,
        seed: int = 42,
        min_tiles: int = 30,
    ) -> None:
        self.dataset = dataset
        self.k_shot = k_shot
        self.min_tiles = min_tiles
        self._rng = random.Random(seed)

        # 预建 class → {scene_id → [tile_idx]} | precompute class → {scene → tiles}
        self._class_scenes: dict[int, dict[int, list[int]]] = {}
        for cls in dataset.visible_classes():
            tiles = dataset.class_to_tiles(cls)
            if len(tiles) < min_tiles:
                continue                       # min_tiles 过滤 | min_tiles filter
            scenes: dict[int, list[int]] = defaultdict(list)
            for idx in tiles:
                scenes[dataset.source_image_id(idx)].append(idx)
            if len(scenes) < 2:
                continue                       # 需至少两个场景以保证不相交 | need ≥2 scenes
            self._class_scenes[cls] = dict(scenes)

        if not self._class_scenes:
            raise ValueError(
                f"No eligible class after filtering (min_tiles={min_tiles}, need ≥2 scenes)."
            )
        self._classes = sorted(self._class_scenes)

    def eligible_classes(self) -> list[int]:
        """通过过滤的类别 ID | class IDs that passed filtering."""
        return list(self._classes)

    def sample(self) -> Episode:
        """采样一个 episode | Sample one episode.

        :return: {"class_id", "support_indices"[≤K], "query_index"}, support 与 query 场景不相交。
        """
        cls = self._rng.choice(self._classes)
        scenes = self._class_scenes[cls]

        # 选一个 query 场景 + 该场景内一张 query tile | pick a query scene + a query tile in it
        query_scene = self._rng.choice(list(scenes))
        query_index = self._rng.choice(scenes[query_scene])

        # support 从其余场景抽取 (与 query 场景不相交) | support drawn from the OTHER scenes
        support_pool = [
            idx for sid, idxs in scenes.items() if sid != query_scene for idx in idxs
        ]
        k = min(self.k_shot, len(support_pool))
        support_indices = self._rng.sample(support_pool, k)

        return {
            "class_id": cls,
            "support_indices": support_indices,
            "query_index": query_index,
        }
