"""
结果记录器：绑定实验 ID 的结果持久化与日志路由。
Results recorder: persistent storage and logging routing tied to experiment ID.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from adasam.config.experiment import ExperimentConfig
from adasam.logging import get_logger
from adasam.logging.backends import ConsoleBackend, FileBackend


class ExperimentRecorder:
    """
    实验结果记录器 | Experiment results recorder.

    绑定实验 ID → 自动创建 output_dir/exp_id/ 目录结构 →
    所有指标通过日志系统路由（Console + File JSONL）。

    Tied to experiment ID → auto-creates output_dir/exp_id/ directory →
    all metrics routed through logging system (Console + File JSONL).

    使用示例 | Usage:
        >>> config = ExperimentConfig(exp_id="exp_test_001", output_dir="./runs")
        >>> recorder = ExperimentRecorder(config)
        >>> recorder.record_metric("miou/val", 0.85, step=100)
        >>> recorder.record_metric("dice/val", 0.78, step=100)
        >>> recorder.finalize()  # 写入 summary.json | Write summary
        >>> recorder.close()     # 关闭所有后端 | Close all backends
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """
        初始化记录器 | Initialize recorder.

        :param config: 实验配置 | Experiment configuration.
        :type config: ExperimentConfig
        """
        self.config = config

        # 实验输出目录 | Experiment output directory
        self._exp_dir = Path(config.output_dir) / config.exp_id
        self._exp_dir.mkdir(parents=True, exist_ok=True)

        # 创建专属 logger + 后端 | Create dedicated logger + backends
        self.logger = get_logger(f"exp.{config.exp_id}")

        # 控制台后端：实时监控 | Console backend: real-time monitoring
        self.logger.add_backend(ConsoleBackend())

        # 文件后端：JSONL 持久化 | File backend: JSONL persistence
        metrics_path = self._exp_dir / "metrics.jsonl"
        self._file_backend = FileBackend(str(metrics_path))
        self.logger.add_backend(self._file_backend)

        # 指标存储（用于 finalize 时写入 summary）| Metric storage (for summary on finalize)
        self._metrics: dict[str, list[float]] = {}

    # ── 指标记录 | Metric Recording ───────────────────────────

    def record_metric(
        self,
        key: str,
        value: float,
        *,
        step: int = 0,
        phase: str = "train",
        tags: Sequence[str] = (),
        **meta: Any,
    ) -> None:
        """
        记录一个指标 | Record a metric.

        同时：1) 通过日志系统路由 | Routes through logging system
              2) 本地累积用于 summary | Accumulates locally for summary

        :param key: 指标名称（如 "miou/val", "dice/val", "fps/infer"）| Metric name.
        :type key: str

        :param value: 指标值 | Metric value.
        :type value: float

        :param step: 全局步数 | Global step.
        :type step: int

        :param phase: 流水线阶段 | Pipeline phase (train/val/test).
        :type phase: str

        :param tags: 标签列表 | Tag list. **meta: 额外元数据 | Additional metadata.
        :type tags: Sequence[str]

        :param meta: 
        :type meta: Any
        """
        self.logger.log_metric(key, value, step=step, phase=phase, tags=tags, **meta)

        # 本地累积 | Local accumulation
        if key not in self._metrics:
            self._metrics[key] = []
        self._metrics[key].append(value)

    def record_config(self) -> None:
        """
        将完整配置写入日志 + YAML 文件 | Log full config + save YAML copy.
        应在训练开始前调用一次。| Should be called once before training starts.
        """
        # 通过 logger 记录每一项配置 | Log each config item via logger
        self.config.log_to_logger(self.logger)

        # 同时保存 YAML 副本 | Also save YAML copy
        yaml_path = self._exp_dir / "config.yaml"
        self.config.to_yaml(str(yaml_path))

        self.logger.log_info("recorder/config_saved", f"Config saved to {yaml_path}")

    # ── Summary 生成 | Summary Generation ──────────────────────

    def finalize(self) -> None:
        """
        写入 summary.json（所有指标的聚合统计）| Write summary.json (aggregated stats).

        包括：
        - exp_id
        - 每个 key 的 min/mean/max
        - 配置快照

        Includes:
        - exp_id
        - min/mean/max per key
        - Config snapshot
        """
        summary: dict[str, Any] = {
            "exp_id": self.config.exp_id,
            "config": self.config.to_dict(),
            "metrics": {},
        }

        # 聚合每个指标 | Aggregate each metric
        for key, values in self._metrics.items():
            if values:
                summary["metrics"][key] = {
                    "count": len(values),
                    "min": min(values),
                    "mean": sum(values) / len(values),
                    "max": max(values),
                    "latest": values[-1],
                }

        # 写入 | Write
        summary_path = self._exp_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        self.logger.log_info(
            "recorder/summary_saved",
            f"Summary saved to {summary_path}",
        )

    # ── 生命周期 | Lifecycle ───────────────────────────────────

    def flush(self) -> None:
        """刷新所有后端缓冲区 | Flush all backend buffers."""
        self.logger.flush()

    def close(self) -> None:
        """
        关闭记录器：刷新并关闭所有后端 | Close recorder: flush and close all backends.
        建议在训练/评测结束时调用。
        Should be called at end of training/evaluation.
        """
        self.flush()
        self.logger.close()
