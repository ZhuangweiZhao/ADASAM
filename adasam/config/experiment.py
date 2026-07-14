"""
实验管理：ID 生成 + 超参数配置 | Experiment management: ID generation + hyperparameter config.
================================================================================================

实验 ID 格式：{prefix}_{name}_{YYYYMMDD}_{HHMMSS}_{mmmm}
Experiment ID format: {prefix}_{name}_{YYYYMMDD}_{HHMMSS}_{mmmm}

其中 mmmm 是微秒后4位，防止同秒内多次调用冲突。
Where mmmm is the last 4 digits of microseconds, preventing same-second collisions.

ExperimentConfig 支持 YAML/JSON 往返，自动通过日志系统记录。
ExperimentConfig supports YAML/JSON round-trip, auto-logged via logging system.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# ── 实验 ID 生成 | Experiment ID Generation ──────────────────


def generate_exp_id(prefix: str = "exp", name: Optional[str] = None) -> str:
    """
    生成唯一的实验 ID | Generate unique experiment ID.

    格式 | Format:
        - 有名称：{prefix}_{name}_{YYYYMMDD}_{HHMMSS}
          With name: {prefix}_{name}_{YYYYMMDD}_{HHMMSS}
        - 无名称：{prefix}_{YYYYMMDD}_{HHMMSS}
          Without name: {prefix}_{YYYYMMDD}_{HHMMSS}

    加入 PID 前缀防止多进程冲突。
    PID suffix prevents collision in multi-process scenarios.

    :param prefix: ID 前缀 | ID prefix (default "exp").
    :type prefix: str

    :param name: 用户自定义名称 | User-specified name. 如果为 None，直接用时间戳。 If None, use timestamp only.
    :type name: Optional[str]

    :return: str: 实验 ID，如 "exp_baseline_20260617_143052" Experiment ID, e.g. "exp_baseline_20260617_143052"
    :rtype: str
    """
    # 时间戳：精确到微秒防止同秒冲突 | Timestamp with microseconds to prevent same-second collision
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    # 微秒部分取后4位，避免同秒内 ID 重复 | Last 4 digits of microseconds for uniqueness within same second
    us = str(now.microsecond)[-4:].zfill(4)

    # 构建 ID | Build ID
    if name:
        # 名称规范化：替换空格和非 ASCII 字符 | Normalize name: replace spaces and non-ASCII
        safe_name = name.replace(" ", "_").lower()
        exp_id = f"{prefix}_{safe_name}_{timestamp}_{us}"
    else:
        exp_id = f"{prefix}_{timestamp}_{us}"

    return exp_id


# ── 实验配置 Dataclass | Experiment Config Dataclass ──────────


@dataclass
class ExperimentConfig:
    """
    实验超参数配置 | Experiment hyperparameter configuration.

    所有训练/评测相关的超参数集中管理。
    Centralized management of all training/evaluation hyperparameters.

    支持与 YAML/JSON 互转；集成日志系统自动记录。
    Supports YAML/JSON round-trip; auto-logged via logging system.

    Fields:
        exp_id:              实验唯一标识 | Unique experiment ID (必需 | required)
        backbone_name:       骨干网络名称 | Backbone model name
        backbone_checkpoint: 预训练权重路径（None = 自动下载）| Pretrained checkpoint path
        image_size:          输入图像尺寸 (H, W) | Input image resolution
        batch_size:          批大小 | Batch size
        learning_rate:       学习率 | Learning rate
        max_epochs:          最大训练轮次 | Maximum training epochs
        seed:                随机种子 | Random seed
        output_dir:          输出根目录 | Output root directory
        dataset_name:        数据集名称 | Dataset name
        dataset_root:        数据集根目录 | Dataset root directory
        num_workers:         数据加载线程数 | DataLoader worker count
    """

    exp_id: str
    # ── 模型配置 | Model config ──
    backbone_name: str = "FastSAM-x"
    backbone_checkpoint: Optional[str] = None
    # ── 训练配置 | Training config ──
    image_size: tuple = (1024, 1024)
    batch_size: int = 1
    learning_rate: float = 1e-4
    max_epochs: int = 50
    seed: int = 42
    # ── 路径配置 | Path config ──
    output_dir: str = "./output"
    dataset_name: str = "iSAID"
    dataset_root: str = "datasets/iSAID"
    num_workers: int = 4

    def __post_init__(self) -> None:
        """
        验证配置合法性 | Validate configuration.
        在 dataclass 构造后自动调用。| Called automatically after dataclass construction.
        """
        # exp_id 不能为空 | exp_id must not be empty
        if not self.exp_id or not self.exp_id.strip():
            raise ValueError(
                f"exp_id 不能为空 | exp_id must not be empty, got: {self.exp_id!r}"
            )

        # batch_size 必须为正整数 | batch_size must be positive
        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size 必须为正整数 | batch_size must be positive, got: {self.batch_size}"
            )

        # learning_rate 必须为正数 | learning_rate must be positive
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate 必须为正数 | learning_rate must be positive, got: {self.learning_rate}"
            )

        # image_size 必须是正整数的二元组 | image_size must be a pair of positive ints
        if len(self.image_size) != 2 or any(s <= 0 for s in self.image_size):
            raise ValueError(
                f"image_size 必须是 (H, W) 正整数组 | image_size must be (H, W) positive ints, "
                f"got: {self.image_size}"
            )

    # ── 序列化 | Serialization ────────────────────────────────

    def to_dict(self) -> dict:
        """
        转为字典 | Convert to dict.
        递归处理嵌套结构（如 image_size tuple → list）。
        Handles nested structures (e.g. image_size tuple → list).
        """
        d = {}
        for key, value in asdict(self).items():
            if isinstance(value, tuple):
                d[key] = list(value)
            elif isinstance(value, Path):
                d[key] = str(value)
            else:
                d[key] = value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """
        从字典创建 | Create from dict.
        自动转换 list → tuple (如 image_size)。
        Auto-converts list → tuple (e.g. image_size).
        """
        # 深拷贝避免修改输入 | Deep copy to avoid mutating input
        kwargs = dict(data)
        # list → tuple 转换 | Convert list fields back to tuples
        if "image_size" in kwargs and isinstance(kwargs["image_size"], list):
            kwargs["image_size"] = tuple(kwargs["image_size"])
        return cls(**kwargs)

    def to_yaml(self, path: str) -> None:
        """
        保存为 YAML 文件 | Save to YAML file.
        自动创建父目录。| Auto-creates parent directory.

        :param path: 输出文件路径 | Output file path.
        :type path: str
        """
        import yaml

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(
                self.to_dict(),
                f,
                default_flow_style=False,
                allow_unicode=True,  # 保留中文 | Preserve Chinese characters
                sort_keys=False,
            )

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """
        从 YAML 文件加载 | Load from YAML file.

        :param path: YAML 文件路径 | Path to YAML file.
        :type path: str

        :return: ExperimentConfig: 新配置实例 | New config instance.
        :rtype: 'ExperimentConfig'
        """
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    # ── 日志集成 | Logging Integration ────────────────────────

    def log_to_logger(self, logger) -> None:
        """
        将所有超参数记录到日志系统 | Log all hyperparameters to the logging system.
        每条配置项作为 INFO 级别记录。
        Each config item logged at INFO level.

        :param logger: adasam.logging.Logger 实例 | Logger instance.
        """
        for key, value in self.to_dict().items():
            logger.log_info(f"config/{key}", str(value))
