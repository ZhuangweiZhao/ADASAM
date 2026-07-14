"""
MetricTracker — 指标时序聚合器 | Aggregates metrics over time.
=================================================================

维护每个 key 独立的运行统计：均值、EMA、最小值、最大值、标准差、计数。
Maintains per-key running statistics: mean, EMA, min, max, std, count.

每个 key 保留自己的滑动窗口历史值。
Each key keeps its own sliding window of historical values.

使用示例 | Usage:
    >>> tracker = MetricTracker(window_size=100, ema_decay=0.99)
    >>> tracker.update("loss/total", 0.5, step=100)
    >>> tracker.update("loss/total", 0.4, step=200)
    >>> stats = tracker.get("loss/total")
    >>> print(stats["mean"], stats["ema"], stats["latest"])
    0.45 0.405 0.4
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


class _MetricState:
    """
    单个指标键的运行统计状态 | Running statistics for a single metric key.

    内部类，由 MetricTracker 管理。| Internal class, managed by MetricTracker.
    """

    __slots__ = (
        "values", "mean", "ema", "min_val", "max_val",
        "count", "sum_val", "_window_size", "_decay",
    )

    def __init__(self, window_size: int = 100, ema_decay: float = 0.99) -> None:
        self.values: list[float] = []         # 滑动窗口 | sliding window
        self.mean: float = 0.0                 # 累计均值 | cumulative mean
        self.ema: float | None = None          # 指数移动平均 | exponential moving average
        self.min_val: float = float("inf")     # 历史最小值 | historical minimum
        self.max_val: float = float("-inf")    # 历史最大值 | historical maximum
        self.count: int = 0                    # 总更新次数 | total update count
        self.sum_val: float = 0.0              # 累计和 | cumulative sum
        self._window_size = window_size
        self._decay = ema_decay

    def update(self, value: float) -> None:
        """
        添加一个值并更新所有统计量 | Add a new value and update all statistics.

        时间复杂度 O(1)，空间复杂度 O(window_size)。
        Time O(1), space O(window_size).
        """
        # 滑动窗口：满了就淘汰最老的 | Sliding window: evict oldest when full
        self.values.append(value)
        if len(self.values) > self._window_size:
            self.values.pop(0)

        # 累计统计 | Cumulative statistics
        self.count += 1
        self.sum_val += value

        # 累计均值 | Cumulative mean
        self.mean = self.sum_val / self.count

        # EMA: ema = decay × old_ema + (1 - decay) × new_value
        # decay 越接近 1 越平滑 | closer to 1 = smoother
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self._decay * self.ema + (1 - self._decay) * value

        # 极值更新 | Min/max update
        if value < self.min_val:
            self.min_val = value
        if value > self.max_val:
            self.max_val = value

    def snapshot(self) -> dict[str, float]:
        """
        返回当前统计快照 | Return current statistics as a dict.

        -------
        dict with keys: mean, ema, min, max, std, count, latest
        """
        return {
            "mean": self.mean,
            "ema": self.ema if self.ema is not None else 0.0,
            "min": self.min_val if self.min_val != float("inf") else 0.0,
            "max": self.max_val if self.max_val != float("-inf") else 0.0,
            "std": self._compute_std(),     # 基于窗口值的标准差 | std of windowed values
            "count": self.count,
            "latest": self.values[-1] if self.values else 0.0,
        }

    def _compute_std(self) -> float:
        """
        计算窗口内值的标准差 | Compute standard deviation of windowed values.
        使用总体标准差（除以 n 而非 n-1）| Uses population std (divides by n).
        """
        if len(self.values) < 2:
            return 0.0
        m = sum(self.values) / len(self.values)
        variance = sum((v - m) ** 2 for v in self.values) / len(self.values)
        return math.sqrt(variance)

    def reset(self) -> None:
        """
        清除所有累积的统计量 | Clear all accumulated statistics.
        """
        self.values.clear()
        self.mean = 0.0
        self.ema = None
        self.min_val = float("inf")
        self.max_val = float("-inf")
        self.count = 0
        self.sum_val = 0.0


class MetricTracker:
    """
    多指标跟踪器 | Multi-key metric tracker.

    维护每个 key 独立的 _MetricState。
    Maintains independent _MetricState for each key.

    线程安全说明：GIL 保护字典操作，无需额外锁。
    Thread-safety: GIL protects dict operations, no extra locks needed.

    ----------
    window_size : int
        每个 key 保留的最近值数量 | Number of recent values to keep per key.
        默认 100。太小 → 标准差不稳定；太大 → 内存占用。
        Default 100. Too small → unstable std; too large → memory overhead.

    ema_decay : float
        EMA 平滑因子 | Smoothing factor for EMA.
        越接近 1 越平滑。默认 0.99 = 约 100 步的有效窗口。
        Closer to 1 = smoother. Default 0.99 ≈ effective window of ~100 steps.
    """

    def __init__(self, window_size: int = 100, ema_decay: float = 0.99) -> None:
        self._window_size = window_size
        self._decay = ema_decay
        # 使用 defaultdict 避免 key 不存在时的 KeyError
        # defaultdict avoids KeyError for unseen keys
        self._states: dict[str, _MetricState] = defaultdict(
            lambda: _MetricState(window_size, ema_decay)
        )

    def update(self, key: str, value: float, *, step: int = 0) -> None:
        """
        为一个 key 添加新值 | Add a value for the given key.
        `step` 只用于引用，不影响统计计算。
        `step` is stored for reference only, not used in stats calculation.
        """
        self._states[key].update(value)

    def get(self, key: str) -> dict[str, float]:
        """
        获取某个 key 的当前统计量 | Get the current statistics for a key.
        如果 key 从未被记录，返回空 dict。| Returns empty dict if key unseen.
        """
        return self._states[key].snapshot()

    def get_value(self, key: str, stat: str = "mean") -> float:
        """
        获取某个 key 的特定统计量 | Get a specific statistic for a key.
        例如 | e.g.: tracker.get_value("loss", "ema") → 指数移动平均

        ----------
        key : str
            指标键名 | Metric key name.
        stat : str
            统计量名：mean, ema, min, max, std, count, latest

        -------
        float: 统计量值，key 不存在返回 0.0 | Returns 0.0 if key unseen.
        """
        return self._states[key].snapshot().get(stat, 0.0)

    @property
    def keys(self) -> list[str]:
        """所有被跟踪的指标键名（排序）| All tracked metric keys (sorted)."""
        return sorted(self._states.keys())

    def snapshot(self) -> dict[str, dict[str, float]]:
        """
        返回完整快照：{key: {mean, ema, min, max, std, count, latest}}.
        Return full snapshot for all tracked keys.
        """
        return {k: s.snapshot() for k, s in self._states.items()}

    def reset(self, key: str | None = None) -> None:
        """
        重置统计量 | Reset statistics.
        - key=None: 重置所有 | reset all
        - key="loss": 只重置该 key | reset only that key
        """
        if key is None:
            self._states.clear()
        else:
            self._states[key].reset()
