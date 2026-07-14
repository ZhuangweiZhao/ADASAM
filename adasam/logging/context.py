"""
LogContext — 日志上下文管理器 | Scoped phase/scope/tag management.
===================================================================

提供上下文管理器 (with 语句) 来管理日志的 phase、scope、step、tags。
Provides context managers (with statements) for log context management.

利用 Python 的 contextvars 实现线程安全/异步安全的上下文栈。
Uses Python's contextvars for thread-safe / async-safe context stack.

这意味着：
- 嵌套 with 语句正确地压栈/弹栈
- 不同线程/协程有各自独立的上下文
- 自动恢复外层上下文

This means:
- Nested with statements correctly push/pop
- Different threads/coroutines have independent contexts
- Outer context is automatically restored

使用示例 | Usage:
    >>> from adasam.logging import get_logger
    >>> logger = get_logger("train")
    >>>
    >>> with logger.phase("val"):
    ...     with logger.scope("decoder"):
    ...         logger.log_metric("iou", 0.85)
    ...         # 这条记录自动携带 phase="val", scope="decoder"
    ...         # This record auto-carries phase="val", scope="decoder"
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Generator, Sequence

# ── 线程安全/异步安全的上下文变量 | Thread-safe / async-safe context variable ─
# contextvars 是 Python 3.7+ 内置的，比 threading.local 更安全
# contextvars (Python 3.7+) is safer than threading.local for async code
_current_context: contextvars.ContextVar[dict[str, object]] = contextvars.ContextVar(
    "log_context",
    default={
        "phase": "train",
        "scope": "",
        "step": 0,
        "tags": (),
    },
)


def get_context() -> dict[str, object]:
    """
    获取当前日志上下文的拷贝 | Get a copy of the current logging context.
    包含 phase, scope, step, tags。
    """
    return _current_context.get().copy()


def get_phase() -> str:
    """获取当前 phase | Get current phase."""
    return str(_current_context.get()["phase"])


def get_scope() -> str:
    """获取当前 scope | Get current scope."""
    return str(_current_context.get()["scope"])


def get_step() -> int:
    """获取当前 step | Get current step."""
    return int(_current_context.get()["step"])


def get_tags() -> tuple[str, ...]:
    """获取当前 tags | Get current tags."""
    return tuple(_current_context.get()["tags"])


class LogContext:
    """
    上下文管理器，临时覆盖日志上下文后自动恢复。
    Context manager that temporarily overrides logging context and auto-restores.

    支持嵌套 — 内层可以覆盖外层的 phase/scope，退出后恢复外层。
    Supports nesting — inner context can override outer, restored on exit.

    tags 行为特殊：内层的 tags 与外层取并集，而非覆盖。
    Special tag behavior: inner tags are UNIONED with outer tags, not replaced.

    ----------
    phase : str | None
        流水线阶段 (train, val, test, infer) | Pipeline phase.
    scope : str | None
        模块范围 (decoder, spm, backbone, loss) | Module scope.
    step : int | None
        全局步数覆盖 | Global step override.
    tags : list[str] | None
        要添加的标签（与外层取并集）| Tags to ADD to the current set.
    """

    def __init__(
        self,
        *,
        phase: str | None = None,
        scope: str | None = None,
        step: int | None = None,
        tags: Sequence[str] | None = None,
    ) -> None:
        self._overrides: dict[str, object] = {}
        if phase is not None:
            self._overrides["phase"] = phase
        if scope is not None:
            self._overrides["scope"] = scope
        if step is not None:
            self._overrides["step"] = step
        if tags is not None:
            self._overrides["tags"] = tags

    def __enter__(self) -> "LogContext":
        # 保存当前上下文用于退出时恢复 | save current context for restore on exit
        ctx = _current_context.get().copy()

        # 应用覆盖值 | Apply overrides
        for k, v in self._overrides.items():
            if k == "tags" and isinstance(v, (list, tuple)):
                # tags 取并集：内层添加不替换 | tags union: inner adds, not replaces
                existing = ctx.get("tags", ())
                ctx["tags"] = tuple(set(existing) | set(v))
            else:
                ctx[k] = v

        self._saved = _current_context.get()
        _current_context.set(ctx)
        return self

    def __exit__(self, *args: object) -> None:
        # 恢复外层上下文 | restore outer context
        _current_context.set(self._saved)


# ── 便捷上下文管理器 | Convenience context managers ──────────
# 提供更简洁的 with 语法
# Simpler with-statement syntax

@contextmanager
def log_phase(phase: str) -> Generator[None, None, None]:
    """临时设置日志 phase | Temporarily set the logging phase."""
    with LogContext(phase=phase):
        yield


@contextmanager
def log_scope(scope: str) -> Generator[None, None, None]:
    """临时设置日志 scope | Temporarily set the logging scope."""
    with LogContext(scope=scope):
        yield


@contextmanager
def log_tags(*tags: str) -> Generator[None, None, None]:
    """临时添加 tags | Temporarily add tags to the logging context."""
    with LogContext(tags=list(tags)):
        yield
