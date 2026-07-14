"""
共享测试夹具 | Shared pytest fixtures.
======================================

从 configs/base.yaml 读取配置与数据根目录, 供数据相关测试使用。
Loads config and data_root from configs/base.yaml for data-dependent tests.
数据缺失时相关测试通过 skipif 跳过 (纯度量/骨干测试不受影响)。
Data-dependent tests skip when the data is absent (metric/backbone tests unaffected).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config() -> dict:
    """解析 configs/base.yaml | parsed configs/base.yaml."""
    with open(_REPO_ROOT / "configs" / "base.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def data_root(config) -> Path:
    """iSAID 数据根目录 | iSAID data root path."""
    return Path(config["data"]["data_root"])
