"""
LogRecord — 日志系统的通用数据容器 | Universal data container for the logging system.
=========================================================================================

每一个可观测值都通过 LogRecord 流动。
Every observable value in the codebase flows through a LogRecord.

这确保了：
- 一致的字段结构 → 方便序列化和过滤
- 自动携带上下文 → phase/scope/step/tags 不会遗漏
- 类型化的日志级别 → 控制台可以根据级别着色

This ensures:
- Consistent field structure → easy serialization and filtering
- Automatic context propagation → phase/scope/step/tags never missing
- Typed log levels → console can color-code by severity
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Sequence


class LogLevel(Enum):
    """
    日志级别 | Log severity / category.

    DEBUG  — 开发调试信息，生产环境可关闭 | Development-only details
    METRIC — 量化指标（损失、IoU、准确率等）| Quantitative measurements
    INFO   — 一般进度消息 | General progress messages
    WARN   — 值得关注的异常 | Anomalies worth attention
    ERROR  — 可恢复的失败 | Recoverable failures
    FATAL  — 不可恢复，训练应停止 | Unrecoverable — training should stop
    """

    DEBUG = auto()
    METRIC = auto()
    INFO = auto()
    WARN = auto()
    ERROR = auto()
    FATAL = auto()


@dataclass(slots=True)
class LogRecord:
    """
    通用日志记录容器 | Universal log container.

    每条记录携带：
    - key:   记录什么（如 "loss/total", "iou/val"） | what is being logged
    - value: 实际数据 | the actual data
    - step:  全局步数（batch 或 epoch） | global step counter
    - phase: 流水线阶段（train, val, test, infer） | pipeline phase
    - scope: 哪个模块产生的（如 "decoder", "spm", "loss"） | which module produced this
    - tags:  任意标签用于过滤（"few-shot", "5-shot", "lambda=5.0"） | arbitrary labels for filtering
    - level: 严重程度 / 类别 | severity / category
    - timestamp: 创建时间戳 | when it was created
    - metadata: 可选的额外上下文字典 | optional dict for extra context
    """

    key: str
    value: Any
    step: int = 0
    phase: str = "train"
    scope: str = ""
    tags: Sequence[str] = ()
    level: LogLevel = LogLevel.METRIC
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── 工厂方法 | Factory helpers ──────────────────────────────
    # 提供语义化构造函数，确保 level 和类型正确
    # Semantic constructors — ensure correct level and type

    @classmethod
    def metric(
        cls,
        key: str,
        value: float,
        *,
        step: int = 0,
        phase: str = "train",
        scope: str = "",
        tags: Sequence[str] = (),
        **meta: Any,
    ) -> "LogRecord":
        """
        创建 METRIC 记录 | Create a METRIC record.
        用于所有量化指标（损失、准确率、IoU 等）。
        For all quantitative values (loss, accuracy, IoU, etc.).
        """
        return cls(
            key=key, value=value, step=step,
            phase=phase, scope=scope, tags=tags,
            level=LogLevel.METRIC, metadata=dict(meta),
        )

    @classmethod
    def info(
        cls,
        key: str,
        value: str,
        *,
        step: int = 0,
        phase: str = "train",
        scope: str = "",
        tags: Sequence[str] = (),
        **meta: Any,
    ) -> "LogRecord":
        """
        创建 INFO 记录 | Create an INFO record.
        用于阶段标记、状态更新等文本消息。
        For phase markers, status updates, etc.
        """
        return cls(
            key=key, value=value, step=step,
            phase=phase, scope=scope, tags=tags,
            level=LogLevel.INFO, metadata=dict(meta),
        )

    @classmethod
    def warn(
        cls,
        key: str,
        value: str,
        *,
        step: int = 0,
        phase: str = "train",
        scope: str = "",
        tags: Sequence[str] = (),
        **meta: Any,
    ) -> "LogRecord":
        """
        创建 WARN 记录 | Create a WARN record.
        用于异常检测、内存警告等。
        For anomaly detection, memory warnings, etc.
        """
        return cls(
            key=key, value=value, step=step,
            phase=phase, scope=scope, tags=tags,
            level=LogLevel.WARN, metadata=dict(meta),
        )

    # ── 序列化 | Serialization ──────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """
        转为 JSON 可序列化字典 | Convert to a JSON-serializable dict.
        用于 FileBackend (JSONL) 和网络传输。
        Used by FileBackend (JSONL) and network transport.
        """
        return {
            "key": self.key,
            "value": self.value,
            "step": self.step,
            "phase": self.phase,
            "scope": self.scope,
            "tags": list(self.tags),
            "level": self.level.name,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"LogRecord(key={self.key!r}, value={self.value!r}, "
            f"step={self.step}, phase={self.phase!r}, scope={self.scope!r}, "
            f"level={self.level.name})"
        )
