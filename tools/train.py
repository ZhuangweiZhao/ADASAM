"""
[DEPRECATED] 训练入口 (Protocol V3 实例分割) | Training entry point (Protocol V3 instance seg).
================================================================================================

**已废弃**: 项目已统一为语义分割。请使用 tools/train_isaid_5i.py。
**Deprecated**: project unified to semantic segmentation. Use tools/train_isaid_5i.py instead.

用法 | Usage::

    python tools/train.py --fold 0 --k-shot 5 --epochs 50 --train-mode novel
    python tools/train.py --epochs 1 --episodes 2            # 冒烟自测 | smoke run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# 使 vendored 之外的 adasam 可导入 (免 editable 安装) | make adasam importable without editable install
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adasam.trainer import Trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AdaSAM few-shot training")
    p.add_argument("--config", default=str(_REPO_ROOT / "configs" / "base.yaml"))
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--k-shot", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--episodes", type=int, default=None, help="episodes per epoch")
    p.add_argument("--train-mode", choices=["base", "novel", "all"], default=None)
    p.add_argument("--data-root", default=None, help="override data.data_root")
    p.add_argument("--train-ann-file", default=None,
                   help="RAM-lean subset annotation JSON (images/folds still from data_root)")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    # ── Dense Prompt Generator ──
    p.add_argument("--num-queries", type=int, default=None,
                   help="[DEPRECATED] FG queries (prompt_generator.num_queries)")
    # ── CAT-SAM Adapter ──
    p.add_argument("--cat-adapter", action="store_true", default=None,
                   help="enable CAT-SAM feature adapter (bottleneck residual conv)")
    return p.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    """加载 yaml 并应用 CLI 覆盖 | Load yaml and apply CLI overrides."""
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 仅在显式提供时覆盖 | override only when explicitly provided
    if args.fold is not None:
        cfg["data"]["fold"] = args.fold
    if args.k_shot is not None:
        cfg["fewshot"]["k_shot"] = args.k_shot
    if args.train_mode is not None:
        cfg["fewshot"]["train_mode"] = args.train_mode
    if args.data_root is not None:
        cfg["data"]["data_root"] = args.data_root
    if args.train_ann_file is not None:
        cfg["data"]["train_ann_file"] = args.train_ann_file
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.episodes is not None:
        cfg["train"]["episodes_per_epoch"] = args.episodes
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.device is not None:
        cfg["train"]["device"] = args.device
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    # ── Dense Prompt Generator ──
    if args.num_queries is not None:
        cfg.setdefault("prompt_generator", {})["num_queries"] = args.num_queries
    # ── CAT-SAM Adapter ──
    if args.cat_adapter is not None:
        cfg["train"]["use_cat_adapter"] = True
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    best = Trainer(cfg).train()
    print(f"[train] done. best checkpoint: {best}")


if __name__ == "__main__":
    main()
