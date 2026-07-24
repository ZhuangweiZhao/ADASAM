"""
iSAID-5i 类别覆盖分析 | Class Coverage Analysis.
==================================================
逐 fold 扫描 train/val，检查每个 base 类是否有 tile 支撑。
Scan each fold's train/val to verify every base class has tiles.

用法 | Usage:
    python tools/analysis/check_folds.py
    python tools/analysis/check_folds.py --data-root data/iSAID-5i
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from adasam.datasets.isaid_5i import ISAID5iDataset, ISAID5I_CATEGORIES, ISAID5I_FOLDS


def check_mode(data_root: str, fold: int, split: str, mode: str, expected: list[int], label: str) -> None:
    """Check a single dataset mode (base or novel) for one split."""
    ds = ISAID5iDataset(root=data_root, fold=fold, split=split, mode=mode)
    visible = sorted(ds.visible_classes())
    missing = sorted(set(expected) - set(visible))

    print(f"  [{split}/{mode}]  tiles={len(ds):>5d}  visible={len(visible)}/{len(expected)}")

    if missing:
        names = [ISAID5I_CATEGORIES.get(c, "?") for c in missing]
        print(f"         >> MISSING: {missing} = {names}")

    stats = ds.class_stats()
    print(f"         {'cls':<5s} {'name':<22s} {'tiles':>6s}  {'coverage':>8s}")
    print(f"         {'---':<5s} {'-'*22} {'-'*6}  {'-'*8}")
    for c in sorted(expected):
        n = stats.get(c, 0)
        name = ISAID5I_CATEGORIES.get(c, "?")
        pct = n / max(len(ds), 1) * 100
        bar = "=" * int(pct / 2) if n > 0 else "(none)"
        flag = "  <<< MISSING" if n == 0 else ""
        print(f"         {c:>5d} {name:<22s} {n:>6d}  {pct:>6.1f}% {bar}{flag}")
    print()


def check_fold(data_root: str, fold: int) -> None:
    base = ISAID5I_FOLDS[fold]["train"]
    novel = ISAID5I_FOLDS[fold]["test"]
    print(f"{'='*70}")
    print(f"  FOLD {fold}")
    print(f"  base  ({len(base)}):  {base}")
    print(f"  novel ({len(novel)}): {novel}")
    print(f"{'='*70}")

    for split in ("train", "val"):
        print(f"  --- {split} ---")
        check_mode(data_root, fold, split, "base", base, "base")
        check_mode(data_root, fold, split, "novel", novel, "novel")

    print()


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="iSAID-5i fold class coverage checker")
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data" / "iSAID-5i"))
    args = p.parse_args()
    data_root = args.data_root

    print(f"\niSAID-5i Class Coverage Audit")
    print(f"data root: {data_root}\n")

    for fold in (0, 1, 2):
        check_fold(data_root, fold)

    # Summary
    print(f"{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for fold in (0, 1, 2):
        fd = ISAID5I_FOLDS[fold]
        base_exp = fd["train"]
        novel_exp = fd["test"]

        tr_base = ISAID5iDataset(root=data_root, fold=fold, split="train", mode="base")
        va_base = ISAID5iDataset(root=data_root, fold=fold, split="val", mode="base")
        tr_novel = ISAID5iDataset(root=data_root, fold=fold, split="train", mode="novel")
        va_novel = ISAID5iDataset(root=data_root, fold=fold, split="val", mode="novel")

        def miss_str(name, exp, visible):
            m = sorted(set(exp) - set(visible))
            if not m:
                return ""
            return f"  {name}_missing={ {c: ISAID5I_CATEGORIES.get(c,'?') for c in m} }"

        print(f"  fold {fold}:")
        print(f"    train/base:  {len(tr_base):>5d} tiles  {len(tr_base.visible_classes())}/{len(base_exp)} classes"
              f"{miss_str('', base_exp, tr_base.visible_classes())}")
        print(f"    val/base:    {len(va_base):>5d} tiles  {len(va_base.visible_classes())}/{len(base_exp)} classes"
              f"{miss_str('', base_exp, va_base.visible_classes())}")
        print(f"    train/novel: {len(tr_novel):>5d} tiles  {len(tr_novel.visible_classes())}/{len(novel_exp)} classes"
              f"{miss_str('', novel_exp, tr_novel.visible_classes())}")
        print(f"    val/novel:   {len(va_novel):>5d} tiles  {len(va_novel.visible_classes())}/{len(novel_exp)} classes"
              f"{miss_str('', novel_exp, va_novel.visible_classes())}")
        print()
    print()


if __name__ == "__main__":
    main()
