"""
[DEPRECATED] adasam.evaluator — 协议 V3 实例分割评估 | Protocol-V3 instance seg evaluation.
=============================================================================================

**已废弃**: 项目已统一为语义分割。请使用 tools/eval_isaid_5i.py。
**Deprecated**: project unified to semantic segmentation. Use tools/eval_isaid_5i.py instead.

此模块仅保留用于参考, 不应在新代码中导入。
This module is kept for reference only; do not import in new code.
"""

import warnings

warnings.warn(
    "adasam.evaluator is deprecated — use tools/eval_isaid_5i.py for semantic segmentation evaluation",
    DeprecationWarning,
    stacklevel=2,
)

from adasam.evaluator.evaluate import Evaluator, build_arg_parser

__all__ = ["Evaluator", "build_arg_parser"]
