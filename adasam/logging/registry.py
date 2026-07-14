"""
LogRegistry — 全局日志注册表 | Global logger registry.
=========================================================

管理命名 Logger 实例及其后端。
Manages named Logger instances and their backends.

主要入口点：get_logger(name) → Logger
Primary entry point: get_logger(name) → Logger

使用示例 | Usage:
    >>> from adasam.logging import get_logger
    >>> from adasam.logging.backends import ConsoleBackend, FileBackend
    >>>
    >>> # 获取/创建 logger | Get or create a logger
    >>> logger = get_logger("train")
    >>> logger.add_backend(ConsoleBackend())
    >>> logger.add_backend(FileBackend("runs/exp1/log.jsonl"))
    >>>
    >>> # 记录指标 | Log a metric
    >>> logger.log_metric("loss/total", 0.5, step=100)
    >>>
    >>> # 切换阶段 | Switch phase
    >>> with logger.phase("val"):
    ...     logger.log_metric("iou", 0.85)
"""

from __future__ import annotations

import atexit
from typing import Any, Sequence

from adasam.logging.base import LogBackend
from adasam.logging.context import LogContext, get_context
from adasam.logging.record import LogLevel, LogRecord
from adasam.logging.tracker import MetricTracker


class Logger:
    """
    命名日志器 | Named logger.

    将记录路由到已注册的后端，同时维护本地 MetricTracker。
    Routes records to registered backends, maintains a local MetricTracker.

    每个 Logger 有自己的 MetricTracker（按模块独立聚合）。
    Each Logger has its own MetricTracker (per-module aggregation).

    Backend 是全局共享的 — 同一个 FileBackend 可以接收来自多个 Logger 的记录。
    Backends are shared globally — one FileBackend can receive from multiple Loggers.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._backends: list[LogBackend] = []
        self._tracker = MetricTracker()

    # ── 后端管理 | Backend management ───────────────────────

    def add_backend(self, backend: LogBackend) -> None:
        """
        注册一个新后端。重复添加安全（幂等）。
        Register a new backend. Duplicate adds are safe (idempotent).
        """
        if backend not in self._backends:
            self._backends.append(backend)

    def remove_backend(self, backend: LogBackend) -> None:
        """移除一个后端 | Remove a backend."""
        if backend in self._backends:
            self._backends.remove(backend)

    @property
    def backends(self) -> list[LogBackend]:
        """当前注册的后端列表 | List of registered backends."""
        return list(self._backends)

    # ── 核心日志方法 | Core logging ─────────────────────────

    def log(self, record: LogRecord) -> None:
        """
        将记录路由到所有已注册的后端 | Route a record to all registered backends.

        自动从当前 LogContext 合并 phase、scope、tags。
        Automatically merges phase, scope, tags from the current LogContext.

        后端写入异常被静默捕获 — 不会让日志失败导致训练崩溃。
        Backend write exceptions are silently caught — log failure won't crash training.
        """
        # 获取当前上下文 | Get current context
        ctx = get_context()

        # 合并上下文默认值：如果 record 字段为默认值，则用上下文填充
        # Merge context defaults: fill from context if record uses default values
        if not record.phase or record.phase == "train":
            record.phase = str(ctx["phase"])
        if not record.scope:
            record.scope = str(ctx["scope"])
        if record.step == 0:
            record.step = int(ctx["step"])
        if not record.tags:
            record.tags = tuple(ctx["tags"])

        # 路由到每个后端 | Route to each backend
        for backend in self._backends:
            try:
                backend.write(record)
            except Exception:
                # 不言不语：日志失败不能影响训练
                # Fail silently: log failure must not affect training
                pass

    # ── 便捷方法 | Convenience methods ──────────────────────

    def log_metric(
        self,
        key: str,
        value: float,
        *,
        step: int = 0,
        phase: str = "",
        scope: str = "",
        tags: Sequence[str] = (),
        **meta: Any,
    ) -> None:
        """
        记录一个量化指标 | Log a numeric metric.

        这是最常用的方法 — 所有标量值都应通过它记录。
        This is the PRIMARY method — all scalar values should go through this.

        同时也更新内部 MetricTracker，方便后续获取聚合统计。
        Also updates the internal MetricTracker for later aggregation queries.
        """
        record = LogRecord.metric(
            key, value, step=step, phase=phase, scope=scope, tags=tags, **meta
        )
        self.log(record)
        # 同时更新本地 tracker，以便后续 snapshot/query
        # Also update local tracker for later snapshot/query
        self._tracker.update(key, value, step=step or int(get_context()["step"]))

    def log_loss(self, key: str, value: float, **kwargs: Any) -> None:
        """
        记录损失值的快捷方法 | Shorthand for log_metric with scope='loss'.
        等价于 log_metric(key, value, scope="loss", ...)。
        """
        self.log_metric(key, value, scope="loss", **kwargs)

    def log_info(self, key: str, value: str, **kwargs: Any) -> None:
        """记录信息消息 | Log an informational message."""
        record = LogRecord.info(key, value, **kwargs)
        self.log(record)

    def log_warn(self, key: str, value: str, **kwargs: Any) -> None:
        """记录警告消息 | Log a warning message."""
        record = LogRecord.warn(key, value, **kwargs)
        self.log(record)

    def log_error(self, key: str, value: str, **kwargs: Any) -> None:
        """记录错误消息 | Log an error message."""
        record = LogRecord(
            key=key, value=value, level=LogLevel.ERROR, **kwargs
        )
        self.log(record)

    # ── 上下文管理器 | Context managers ─────────────────────

    def context(
        self,
        *,
        phase: str | None = None,
        scope: str | None = None,
        step: int | None = None,
        tags: Sequence[str] | None = None,
    ) -> LogContext:
        """
        创建作用域上下文管理器 | Create a scoped context manager.

        使用 with 语句包裹代码块。| Use with 'with' statement.

        Example:
            >>> with logger.context(phase="val", scope="decoder"):
            ...     logger.log_metric("iou", 0.85)
        """
        return LogContext(phase=phase, scope=scope, step=step, tags=tags)

    def phase(self, name: str) -> LogContext:
        """
        快捷方法：切换 phase | Shorthand for context(phase=name).

        Example:
            >>> with logger.phase("val"):
            ...     logger.log_metric("iou", 0.85)
        """
        return LogContext(phase=name)

    def scope(self, name: str) -> LogContext:
        """
        快捷方法：设置 scope | Shorthand for context(scope=name).

        Example:
            >>> with logger.scope("decoder"):
            ...     logger.log_metric("iou", 0.85)
        """
        return LogContext(scope=name)

    # ── Tracker 访问 | Tracker access ───────────────────────

    @property
    def tracker(self) -> MetricTracker:
        """
        内部 MetricTracker。
        Internal MetricTracker for aggregated statistics.
        """
        return self._tracker

    def get_stats(self, key: str) -> dict[str, float]:
        """
        获取某个指标的聚合统计 | Get aggregated stats for a metric key.
        等价于 self.tracker.get(key)。
        """
        return self._tracker.get(key)

    @property
    def summary(self) -> dict[str, dict[str, float]]:
        """
        所有被跟踪指标的完整快照 | Full snapshot of all tracked metrics.
        返回 {key: {mean, ema, min, max, std, count, latest}}。
        """
        return self._tracker.snapshot()

    # ── 生命周期 | Lifecycle ───────────────────────────────

    def flush(self) -> None:
        """刷新所有后端的缓冲区 | Flush all backends' buffers."""
        for backend in self._backends:
            backend.flush()

    def close(self) -> None:
        """关闭所有后端，释放资源 | Close all backends, release resources."""
        for backend in self._backends:
            backend.close()
        self._backends.clear()


class LogRegistry:
    """
    全局单例注册表 | Global singleton registry.

    管理所有命名 Logger 实例。
    Manages all named Logger instances.

    通常通过 get_logger() 访问，不直接使用。
    Typically accessed via get_logger(), not directly.
    """

    _instance: "LogRegistry | None" = None

    def __init__(self) -> None:
        self._loggers: dict[str, Logger] = {}

    @classmethod
    def instance(cls) -> "LogRegistry":
        """获取全局单例 | Get the global singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get(self, name: str) -> Logger:
        """
        通过名称获取或创建 Logger | Get or create a Logger by name.
        相同名称返回同一个 Logger 实例（单例模式）。
        Same name returns the same Logger instance (singleton pattern).
        """
        if name not in self._loggers:
            self._loggers[name] = Logger(name)
        return self._loggers[name]

    def list_loggers(self) -> list[str]:
        """列出所有已注册的 Logger 名称 | Return names of all registered loggers."""
        return sorted(self._loggers.keys())

    def close_all(self) -> None:
        """
        关闭所有 Logger 及其后端 | Close all loggers and their backends.
        在程序退出时调用。| Called at program exit.
        """
        for logger in self._loggers.values():
            logger.close()
        self._loggers.clear()


# ── 公共 API | Public API ──────────────────────────────────


def get_logger(name: str = "default") -> Logger:
    """
    获取或创建一个命名的 Logger | Get or create a named Logger.

    这是日志系统的**唯一入口点**。
    This is the **single entry point** for the logging system.

    ----------
    name : str
        Logger 名称。建议按子系统命名：
        - "train"   — 训练循环 | training loop
        - "data"    — 数据集/数据加载 | dataset/data loading
        - "model"   — 模型内部 | model internals

    -------
    Logger
        命名 Logger 实例 | The named Logger instance.

    Examples
    --------
    >>> # 创建训练 logger 并添加后端
    >>> logger = get_logger("train")
    >>> logger.add_backend(ConsoleBackend())
    >>>
    >>> # 记录训练指标
    >>> logger.log_loss("total", 0.5, step=100)
    >>>
    >>> # 切换验证阶段
    >>> with logger.phase("val"):
    ...     logger.log_metric("iou", 0.85)
    >>>
    >>> # 获取聚合统计
    >>> print(logger.summary)
    {'loss/total': {'mean': 0.5, 'ema': 0.5, ...}}
    """
    return LogRegistry.instance().get(name)


# ── 自动清理 | Auto-cleanup ─────────────────────────────────
# 确保 Python 退出时所有后端优雅关闭
# Ensure all backends are gracefully closed on interpreter exit

@atexit.register
def _cleanup() -> None:
    """解释器退出时自动关闭所有后端 | Auto-close all backends on interpreter exit."""
    LogRegistry.instance().close_all()
