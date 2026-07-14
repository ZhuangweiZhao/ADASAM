"""
adasam.logging — 结构化日志系统 | Structured Logging System
=============================================================

日志系统是整个项目第一个构建的核心基础设施模块。
所有后续代码必须通过此系统路由所有可观测值。
The logging system is the FIRST and CORE infrastructure module.
All subsequent code MUST route values through this system.

架构总览 | Architecture:
    LogRecord   — 通用数据容器，记录"要记什么" | universal data container (what to log)
    LogBackend  — 抽象输出后端，决定"写到哪里" | where to write (console, file, wandb, tensorboard)
    LogContext  — 上下文管理器，标注"在哪个阶段/范围" | when/where in the pipeline (phase, scope, step)
    MetricTracker — 时序聚合器，计算运行均值/EMA/标准差 | aggregation over time (running mean, EMA, etc.)
    LogRegistry — 全局单例注册表，通过名称获取 Logger | global singleton registry (get_logger by name)

设计原则 | Design principles:
1. 每个可观测值都走 Logger——禁止裸 print()
   EVERY value worth observing goes through the logger — no bare print()
2. 结构化：每条记录携带 step, phase, scope, tags
   Structured: all records carry step, phase, scope, tags
3. 可插拔后端：增减输出 sink 不改变业务代码
   Pluggable backends: add/remove sinks without changing code
4. 非阻塞：文件 I/O 在后台线程执行，不拖慢训练
   Non-blocking: file I/O offloaded to background thread

使用示例 | Usage:
    >>> from adasam.logging import get_logger
    >>> logger = get_logger("train")
    >>> logger.add_backend(ConsoleBackend())
    >>> logger.add_backend(FileBackend("runs/exp1/log.jsonl"))
    >>>
    >>> with logger.phase("train"):
    ...     logger.log_metric("loss/total", 0.5, step=100)
    ...     logger.log_loss("seg", 0.3, step=100)
"""

from adasam.logging.record import LogRecord, LogLevel
from adasam.logging.base import LogBackend
from adasam.logging.tracker import MetricTracker
from adasam.logging.context import LogContext
from adasam.logging.registry import LogRegistry, get_logger

__all__ = [
    "LogRecord",
    "LogLevel",
    "LogBackend",
    "MetricTracker",
    "LogContext",
    "LogRegistry",
    "get_logger",
]
