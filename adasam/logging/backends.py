"""
内置日志后端 | Built-in log backends: Console, File (JSONL), Wandb, TensorBoard.
================================================================================

ConsoleBackend  — 彩色终端输出，实时监控训练进度
                  Colorized console output for real-time monitoring.

FileBackend     — JSONL 文件写入（后台线程非阻塞），持久化每一条记录
                  JSONL file writing (non-blocking background thread), persists every record.

WandbBackend    — Weights & Biases 集成，自动 log 标量指标
                  W&B integration, auto-logs scalar metrics.

使用示例 | Usage:
    >>> from adasam.logging.backends import ConsoleBackend, FileBackend
    >>> from adasam.logging import get_logger
    >>>
    >>> logger = get_logger("train")
    >>> logger.add_backend(ConsoleBackend())               # 终端实时监控 | real-time console
    >>> logger.add_backend(FileBackend("runs/exp1/log.jsonl"))  # 持久化 | persistence
"""

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path
from typing import TextIO

from adasam.logging.base import LogBackend
from adasam.logging.record import LogLevel, LogRecord


# ── Console Backend | 控制台后端 ────────────────────────────────
# 彩色输出，不同级别不同颜色，一目了然
# Colorized output — different colors per level for quick scanning

_COLOR_MAP: dict[LogLevel, str] = {
    LogLevel.DEBUG:  "\033[90m",   # 灰色 grey
    LogLevel.METRIC: "\033[36m",   # 青色 cyan
    LogLevel.INFO:   "\033[0m",    # 默认 default
    LogLevel.WARN:   "\033[33m",   # 黄色 yellow
    LogLevel.ERROR:  "\033[31m",   # 红色 red
    LogLevel.FATAL:  "\033[1;31m", # 粗体红色 bold red
}
_RESET = "\033[0m"  # ANSI 重置码 | ANSI reset code


class ConsoleBackend(LogBackend):
    """
    终端彩色输出后端 | Pretty-prints log records to stdout.

    输出格式:
        [  step] [       PHASE:SCOPE] key = value   (tags...)

    不同日志级别使用不同 ANSI 颜色代码。
    Different log levels use different ANSI color codes.

    ----------
    min_level : str
        最低输出级别 | Minimum output level (default "DEBUG").
    use_color : bool
        是否使用 ANSI 颜色 | Whether to use ANSI color codes.
        在 Windows 终端不支持 ANSI 时设为 False。
        Set to False on terminals without ANSI support.
    """

    def __init__(self, *, min_level: str = "DEBUG", use_color: bool = True) -> None:
        super().__init__(min_level=min_level)
        self._use_color = use_color

    def write(self, record: LogRecord) -> None:
        if not self.should_accept(record):
            return

        color = _COLOR_MAP.get(record.level, "") if self._use_color else ""
        # scope 为空时不显示冒号 | omit colon when scope is empty
        scope_str = f"{record.phase}:{record.scope}" if record.scope else record.phase
        # tags 非空时显示在行尾 | show tags at end of line when present
        tags_str = f"  ({', '.join(record.tags)})" if record.tags else ""

        # step 定宽 6 位右对齐，phase:scope 自适应宽度 | step fixed 6-char right-align, phase:scope adaptive
        line = (
            f"{color}"
            f"[{record.step:>6d}] [{scope_str}] "
            f"{record.key} = {record.value}"
            f"{tags_str}"
            f"{_RESET if self._use_color else ''}"
        )
        print(line)


# ── File Backend | 文件后端 (JSONL) ─────────────────────────────
# 后台线程写入，不阻塞训练主循环
# Background thread writing — never blocks the training loop

class FileBackend(LogBackend):
    """
    JSONL 文件后端 | Writes log records to a JSONL file.

    特点 | Features:
    - 非阻塞：写入在后台线程执行 | Non-blocking: writes happen on a background thread
    - 自动创建父目录 | Auto-creates parent directories
    - 每行一个 JSON 对象（JSONL 格式），可直接用 pandas 读取
      One JSON object per line (JSONL), directly readable by pandas
    - 文件追加模式，每次运行写入同一个文件
      Append mode — each run appends to the same file

    ----------
    filepath : str | Path
        输出文件路径 | Output file path.
    min_level : str
        最低输出级别 | Minimum output level.
    buffer_size : int
        批量写入大小（攒够此数量才 flush） | Batch size before flush.
    flush_interval : float
        强制刷新间隔（秒），即使不满 buffer 也 flush
        Force flush interval in seconds.
    """

    def __init__(
        self,
        filepath: str | Path,
        *,
        min_level: str = "DEBUG",
        buffer_size: int = 1,
        flush_interval: float = 1.0,
    ) -> None:
        super().__init__(min_level=min_level)
        self._path = Path(filepath)
        # 自动创建父目录 | auto-create parent directories
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval

        self._fh: TextIO | None = None
        # 使用 queue.Queue 实现线程安全的生产者-消费者
        # queue.Queue for thread-safe producer-consumer pattern
        self._queue: queue.Queue[LogRecord | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = True

        self._open()
        self._start_thread()

    # ── 生命周期管理 | Lifecycle ─────────────────────────────

    def _open(self) -> None:
        """打开文件句柄（追加模式，行缓冲）| Open the file handle (append mode, line buffered)."""
        self._fh = open(self._path, "a", encoding="utf-8", buffering=1)

    def _start_thread(self) -> None:
        """启动后台写入线程 | Launch the background writer thread."""
        self._thread = threading.Thread(
            target=self._writer_loop,
            name=f"log-writer-{self._path.stem}",
            daemon=True,  # 守护线程：主进程退出时自动终止 | daemon: auto-kill on main exit
        )
        self._thread.start()

    def _writer_loop(self) -> None:
        """
        后台循环：从队列取记录 → 攒批 → 写盘
        Background loop: drain the queue and write to disk in batches.
        """
        batch: list[LogRecord] = []

        while self._running:
            try:
                # 阻塞等待新记录，超时则强制 flush
                # Block until new record arrives, or timeout → force flush
                record = self._queue.get(timeout=self._flush_interval)
            except queue.Empty:
                # 超时：刷新缓冲区，防止长时间无日志时数据丢失
                # Timeout: flush buffer to prevent data loss during quiet periods
                if batch and self._fh is not None:
                    self._fh.flush()
                continue

            try:
                if record is None:  # sentinel 信号 → 停止 | sentinel — stop
                    self._flush_batch(batch)
                    batch.clear()  # 清空防止 flush_batch 后的重复写入 | clear to prevent double-write
                    break

                batch.append(record)
                # 攒够 buffer_size 就批量写出 | flush when batch is full
                if len(batch) >= self._buffer_size:
                    self._flush_batch(batch)
                    batch.clear()
            finally:
                # 标记队列任务完成（用于 queue.join() 同步）
                # Mark task done (for queue.join() synchronization)
                self._queue.task_done()

    def _flush_batch(self, batch: list[LogRecord]) -> None:
        """
        批量写出记录到文件 | Write a batch of records to the file.
        每条记录一行 JSON | One JSON line per record.
        """
        if not batch or self._fh is None:
            return
        for record in batch:
            # ensure_ascii=False 保留中文等非 ASCII 字符
            # ensure_ascii=False preserves non-ASCII characters (e.g. Chinese)
            line = json.dumps(record.to_dict(), ensure_ascii=False)
            self._fh.write(line + "\n")
        self._fh.flush()

    # ── 后端接口实现 | Backend interface ─────────────────────

    def write(self, record: LogRecord) -> None:
        """将记录放入队列供后台线程写入 | Enqueue a record for background writing."""
        if not self.should_accept(record):
            return
        if self._running:
            self._queue.put(record)

    def flush(self) -> None:
        """阻塞直到队列排空 | Block until the queue is drained."""
        self._queue.join()

    def close(self) -> None:
        """
        优雅关闭：排空队列 → 发送停止信号 → 等待线程退出 → 关闭文件
        Graceful shutdown: drain queue → send stop signal → join thread → close file.
        """
        if not self._running:
            return
        # 步骤1：等待队列中所有项目被处理（排空）
        # Step 1: wait for all queued items to be processed
        # Step 1: send stop sentinel FIRST to wake writer from blocking get()
        # (avoids deadlock: writer blocked on get() while close() blocks on join())
        self._queue.put(None)
        # Step 2: wait for all items (including sentinel) to be processed
        self._queue.join()
        # Step 3: mark stopped
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        # 步骤4：关闭文件句柄 | Step 4: close the file handle
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# ── Wandb Backend | Weights & Biases 后端 ──────────────────────
# 可选依赖 — 仅在 import wandb 成功时可用
# Optional dependency — only available if wandb is installed

class WandbBackend(LogBackend):
    """
    W&B 后端 | Routes log records to Weights & Biases.

    标量 METRIC → wandb.log()
    其他记录  → wandb.summary / notes

    ----------
    project : str
        W&B 项目名 | W&B project name.
    name : str | None
        运行名称 | Run name (auto-generated if None).
    config : dict | None
        超参数配置 | Hyperparameter config for W&B dashboard.
    min_level : str
        最低输出级别 | Minimum output level (default "METRIC" to avoid debug spam).
    **wandb_kwargs
        其他 wandb.init() 参数 | Additional wandb.init() kwargs.
    """

    def __init__(
        self,
        *,
        project: str = "adasam",
        name: str | None = None,
        config: dict | None = None,
        min_level: str = "METRIC",
        **wandb_kwargs,
    ) -> None:
        super().__init__(min_level=min_level)
        self._project = project
        self._name = name
        self._config = config
        self._wandb_kwargs = wandb_kwargs
        self._initialized = False
        self._run = None

    def _lazy_init(self) -> None:
        """
        延迟初始化：第一次 write() 时才 import wandb 并 init。
        Lazy init: defer wandb import + init until first write().
        这样即使没安装 wandb 也不影响其他后端的使用。
        This allows other backends to work even without wandb installed.
        """
        if self._initialized:
            return
        try:
            import wandb
        except ImportError:
            # wandb 未安装，标记为已初始化避免反复尝试
            # wandb not installed — mark initialized to prevent retry
            self._initialized = True
            return

        self._run = wandb.init(
            project=self._project,
            name=self._name,
            config=self._config,
            **self._wandb_kwargs,
        )
        self._initialized = True

    def write(self, record: LogRecord) -> None:
        if not self.should_accept(record):
            return
        self._lazy_init()
        if self._run is None:
            return

        # 标量指标 → wandb.log() | Scalar metrics → wandb.log()
        if record.level == LogLevel.METRIC and isinstance(record.value, (int, float)):
            self._run.log(
                {record.key: record.value},
                step=record.step,
            )
        # 警告/错误 → wandb alert (未来实现) | Warnings/errors → wandb alert (TODO)
        elif record.level in (LogLevel.WARN, LogLevel.ERROR):
            pass

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._initialized = False
            self._run = None
