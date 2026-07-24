"""adasam.datasets — 语义分割数据集与 episode 采样 | Semantic segmentation data & episode sampling."""

# [DEPRECATED] Legacy instance dataset — no longer imported.
# ISAID_CATEGORIES / DEFAULT_FOLDS are kept as re-exports for backward-compat with
# any scripts that still reference them via adasam.datasets.
# These constants are also available from adasam.datasets.isaid_5i (ISAID5I_CATEGORIES, etc.).
from adasam.datasets.isaid import (
    ISAID_CATEGORIES,
    DEFAULT_FOLDS,
)
# [DEPRECATED] MIN_INSTANCE_AREA → renamed to MIN_REGION_AREA in isaid_5i
from adasam.datasets.isaid import MIN_INSTANCE_AREA  # noqa: F401 — backward compat
from adasam.datasets.neu_seg import (
    NEUSegDataset,
    NEUSEG_CLASS_ID,
    NEUSEG_CLASS_NAME,
    NEUSEG_CATEGORIES,
)
from adasam.datasets.isaid_5i import (
    ISAID5iDataset,
    ISAID5iEpisodeSampler,
    ISAID5I_CATEGORIES,
    ISAID5I_FOLDS,
)
from adasam.datasets.episode import EpisodeSampler, Episode

__all__ = [
    "ISAID_CATEGORIES",
    "DEFAULT_FOLDS",
    "NEUSegDataset",
    "NEUSEG_CLASS_ID",
    "NEUSEG_CLASS_NAME",
    "NEUSEG_CATEGORIES",
    "ISAID5iDataset",
    "ISAID5iEpisodeSampler",
    "ISAID5I_CATEGORIES",
    "ISAID5I_FOLDS",
    "EpisodeSampler",
    "Episode",
]
