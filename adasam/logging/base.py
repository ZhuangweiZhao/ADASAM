"""
LogBackend — 日志输出后端的抽象基类 | Abstract base class for log sinks.
============================================================================

所有后端（Console, File, Wandb, TensorBoard）都继承此类。
All backends inherit from this base.

子类只需要重写 write() 方法即可接入日志系统。
Subclasses only need to override write() to plug into the system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from adasam.logging.record import LogRecord


class LogBackend(ABC):
    """
    日志记录的抽象输出端 | Abstract sink for log records.

    子类化并重写 write() 将记录发送到目标位置。
    Subclass and override write() to send records to a destination.

    ----------
    min_level : str
        最低输出级别，低于此级别的记录被丢弃 | Minimum level to output; records below are dropped.
        可选值 | Options: "DEBUG", "METRIC", "INFO", "WARN", "ERROR", "FATAL"
        默认 | Default: "DEBUG" (输出所有 | output all)
    """

    def __init__(self, *, min_level: str = "DEBUG") -> None:
        from adasam.logging.record import LogLevel

        self.min_level = LogLevel[min_level.upper()]

    @abstractmethod
    def write(self, record: LogRecord) -> None:
        """
        向此后端写入一条记录 | Write a single record to this backend.

        子类必须实现此方法。| Subclasses MUST implement this.
        """
        ...

    def flush(self) -> None:
        """
        刷新缓冲区 | Flush any buffered data.
        如果后端有缓冲（如 FileBackend），重写此方法。
        Override if the backend buffers data.
        """
        pass

    def close(self) -> None:
        """
        关闭后端，释放资源 | Close the backend, releasing resources.
        在训练结束或程序退出时调用。
        Called when training ends or program exits.
        """
        pass

    def should_accept(self, record: LogRecord) -> bool:
        """
        判断是否接受此记录 | Return True if this backend should accept the record.
        根据 min_level 过滤。| Filter based on min_level.
        """
        return record.level.value >= self.min_level.value
