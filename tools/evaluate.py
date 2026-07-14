"""
评估入口 | Evaluation entry point.
==================================

AdaSAM 唯一评估脚本 (协议 V3)。所有论文数字必须来自此脚本。
The single evaluation script (Protocol V3). All paper numbers must come from here.

用法 | Usage::

    python tools/evaluate.py --checkpoint runs/.../best_model.pt --k-shot 5 --seed 42
    python tools/evaluate.py --checkpoint <ckpt> --limit 5 --no-zero-shot   # 冒烟 | smoke
"""

from __future__ import annotations

import sys
from pathlib import Path

# 使 adasam 可导入 (免 editable 安装) | make adasam importable without editable install
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adasam.evaluator import Evaluator, build_arg_parser  # noqa: E402


def main() -> None:
    args = build_arg_parser().parse_args()
    result = Evaluator(args).run()
    ft = result["finetuned"]
    print(f"[eval] AP={ft['AP']:.4f} AP50={ft['AP50']:.4f} "
          f"InstMIoU={ft['instance_miou_overall']:.4f}")


if __name__ == "__main__":
    main()
