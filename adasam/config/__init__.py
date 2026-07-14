"""
adasam.config — 实验管理与配置 | Experiment Management & Configuration.
=========================================================================

导出 | Exports:
    generate_exp_id()    — 生成唯一实验 ID | Generate unique experiment ID
    ExperimentConfig     — 超参数配置 dataclass | Hyperparameter config dataclass
    ExperimentRecorder   — 结果记录器（绑定 exp_id）| Results recorder (tied to exp_id)
"""

from adasam.config.experiment import ExperimentConfig, generate_exp_id
from adasam.config.recorder import ExperimentRecorder

__all__ = [
    "generate_exp_id",
    "ExperimentConfig",
    "ExperimentRecorder",
]
