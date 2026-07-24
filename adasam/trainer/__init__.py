"""
[DEPRECATED] adasam.trainer — Protocol V3 实例分割训练 | Protocol-V3 instance seg training.
=============================================================================================

**已废弃**: 项目已统一为语义分割。请使用 tools/train_isaid_5i.py。
**Deprecated**: project unified to semantic segmentation. Use tools/train_isaid_5i.py instead.

此模块仅保留用于参考, 不应在新代码中导入。
This module is kept for reference only; do not import in new code.
"""

import warnings

warnings.warn(
    "adasam.trainer is deprecated — use tools/train_isaid_5i.py for semantic segmentation training",
    DeprecationWarning,
    stacklevel=2,
)

from adasam.trainer.trainer import Trainer

__all__ = ["Trainer"]
