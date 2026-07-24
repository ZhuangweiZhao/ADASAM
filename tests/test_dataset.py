"""
[DEPRECATED] iSAID 遗留实例数据集单元测试 | Unit tests for legacy ISAIDInstanceDataset.
=========================================================================================

覆盖 | Covers:
    - Base/Novel 可见类划分 | base/novel visible-class split.
    - 样本契约: 键、形状、dtype、取值范围 | sample contract: keys, shapes, dtype, ranges.
    - 查询接口: class_to_tiles / source_image_id | query interface.

依赖真实数据 (configs/base.yaml:data_root); 缺失时整文件跳过。
Depends on real data (configs/base.yaml:data_root); the whole file skips if absent.

NOTE: ISAIDInstanceDataset 已废弃, 项目统一为 ISAID5iDataset (语义分割).
      ISAIDInstanceDataset is deprecated; project unified to ISAID5iDataset (semantic).
"""

from __future__ import annotations

import pytest
import torch

from adasam.datasets.isaid import ISAIDInstanceDataset
from adasam.datasets import DEFAULT_FOLDS


@pytest.fixture(scope="module")
def _require_data(data_root):
    if not (data_root / "annotations" / "instances_val.json").exists():
        pytest.skip(f"iSAID data not found at {data_root}")
    return data_root


@pytest.fixture(scope="module")
def ds_base(_require_data):
    # val split 较小, 加载更快 | val split is smaller / faster to load.
    return ISAIDInstanceDataset(_require_data, split="val", fold=0, mode="base")


def test_base_visible_classes(ds_base):
    """base 模式可见类 == fold0 base 定义 | base visible classes match fold-0 base."""
    assert ds_base.visible_classes() == sorted(DEFAULT_FOLDS[0]["base"])
    assert len(ds_base) > 0


def test_sample_contract(ds_base):
    """样本键/形状/dtype/取值范围 | sample keys, shapes, dtype, value ranges."""
    s = ds_base[0]
    assert set(s.keys()) == {"image", "regions", "tile_id", "orig_image_id", "tile_origin"}

    img = s["image"]
    assert img.shape == (3, 896, 896)
    assert img.dtype == torch.float32
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0

    assert isinstance(s["regions"], list)
    assert isinstance(s["tile_id"], int)
    assert isinstance(s["orig_image_id"], int)
    assert isinstance(s["tile_origin"], tuple) and len(s["tile_origin"]) == 2


def test_instance_fields(_require_data):
    """实例字段: mask bool[896,896], 类别∈可见, bbox 长度4, area>0 | per-instance field contract."""
    ds = ISAIDInstanceDataset(_require_data, split="val", fold=0, mode="base")
    visible = set(ds.visible_classes())
    # 找到一个含实例的 tile | find a tile that actually has instances
    inst = None
    for i in range(min(len(ds), 200)):
        s = ds[i]
        if s["regions"]:
            inst = s["regions"][0]
            break
    assert inst is not None, "no tile with instances found in first 200 tiles"
    assert inst["mask"].shape == (896, 896) and inst["mask"].dtype == torch.bool
    assert inst["category_id"] in visible
    assert len(inst["bbox"]) == 4
    assert inst["area"] > 0


def test_instances_restricted_to_visible(ds_base):
    """样本内所有实例类别都在可见集合内 | all instance categories are within the visible set."""
    visible = set(ds_base.visible_classes())
    for i in range(min(len(ds_base), 50)):
        for inst in ds_base[i]["regions"]:
            assert inst["category_id"] in visible


def test_novel_mode_visible_classes(_require_data):
    """novel 模式可见类 == fold0 novel 定义 | novel visible classes match fold-0 novel."""
    ds = ISAIDInstanceDataset(_require_data, split="val", fold=0, mode="novel")
    assert ds.visible_classes() == sorted(DEFAULT_FOLDS[0]["novel"])


def test_query_interface(ds_base):
    """class_to_tiles 返回有效索引, source_image_id 可用 | query interface returns valid data."""
    cls = ds_base.visible_classes()[0]
    tiles = ds_base.class_to_tiles(cls)
    assert isinstance(tiles, list) and len(tiles) > 0
    # class_to_tiles 索引处的 tile 确实含该类 | tiles listed for a class truly contain it
    idx = tiles[0]
    cats = {inst["category_id"] for inst in ds_base[idx]["regions"]}
    assert cls in cats
    assert isinstance(ds_base.source_image_id(idx), int)
