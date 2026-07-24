"""
构建 RAM 精简的训练子集 | Build a RAM-lean training subset.
============================================================

用 ijson 常量内存流式过滤 iSAID train COCO JSON, 只保留目标类 (默认某 fold 的 novel 类) 且每类
至多 N 张 tile, 生成一个可在低内存机器上加载的小 JSON。图像仍读自原始目录 (不复制)。
Streams the iSAID train COCO JSON with constant memory (ijson), keeping only target classes
(default: a fold's novel classes) and at most N tiles per class, producing a small JSON loadable
on a low-RAM machine. Images are still read from the original directory (not copied).

用法 | Usage::

    python tools/analysis/make_subset.py \
        --src  <data_root>/annotations/instances_train.json \
        --dst  data/subsets/instances_train_novel_fold0.json \
        --fold 0 --max-tiles-per-class 200
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

import ijson

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adasam.datasets import DEFAULT_FOLDS  # noqa: E402


def _dec(o):
    """json.dump 兜底: Decimal → int/float | serialize ijson Decimals."""
    if isinstance(o, Decimal):
        return int(o) if o == o.to_integral_value() else float(o)
    raise TypeError(f"not serializable: {type(o).__name__}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a RAM-lean novel-class training subset")
    p.add_argument("--src", required=True, help="source instances_train.json")
    p.add_argument("--dst", required=True, help="destination subset JSON")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--classes", type=int, nargs="*", default=None,
                   help="target class ids (default: fold novel classes)")
    p.add_argument("--max-tiles-per-class", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    targets = set(args.classes) if args.classes else set(DEFAULT_FOLDS[args.fold]["novel"])
    print(f"[subset] targets={sorted(targets)} max_tiles/class={args.max_tiles_per_class}")

    # ── pass 1: 每类的 image_id 集合 (仅 id, 常量内存) | per-class image-id sets ──
    cat_to_imgs: dict[int, set[int]] = {c: set() for c in targets}
    with open(src, "rb") as f:
        for ann in ijson.items(f, "annotations.item"):
            cat = int(ann["category_id"])
            if cat in targets:
                cat_to_imgs[cat].add(int(ann["image_id"]))
    for c in sorted(targets):
        print(f"[subset]   class {c}: {len(cat_to_imgs[c])} tiles available")

    # ── 每类选前 N 张 (排序确定性) → 保留图像集合 | pick ≤N tiles per class (deterministic) ──
    keep_images: set[int] = set()
    for c in sorted(targets):
        keep_images.update(sorted(cat_to_imgs[c])[: args.max_tiles_per_class])
    print(f"[subset] keeping {len(keep_images)} unique tiles")

    # ── pass 2: 保留的图像 | kept images ──
    images = []
    with open(src, "rb") as f:
        for img in ijson.items(f, "images.item"):
            if int(img["id"]) in keep_images:
                images.append(img)

    # ── pass 3: 保留图像内的目标类标注 | target-class annotations within kept images ──
    annotations = []
    with open(src, "rb") as f:
        for ann in ijson.items(f, "annotations.item"):
            if int(ann["image_id"]) in keep_images and int(ann["category_id"]) in targets:
                annotations.append(ann)

    # ── categories (全部, 很小) | all categories (small) ──
    with open(src, "rb") as f:
        categories = list(ijson.items(f, "categories.item"))

    out = {"images": images, "annotations": annotations, "categories": categories}
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, default=_dec)
    size_mb = dst.stat().st_size / 1e6
    print(f"[subset] wrote {len(images)} images, {len(annotations)} annotations "
          f"→ {dst} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
