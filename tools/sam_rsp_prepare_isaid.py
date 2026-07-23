"""
SAM-RSP iSAID-5i 数据准备 | Data Preparation for SAM-RSP on iSAID-5i.
=====================================================================

扫描 iSAID-5i 所有 tile, 按 fold 建立:
  1. class→tile 映射 (用于 few-shot episode 采样)
  2. 图像-标签对列表 (data_list)
  3. Base class tiles (用于 PSPNet 预训练)

iSAID-5i fold 定义 (15 类, 3-fold):
  Fold 0: base={6-15}, novel={1-5}
  Fold 1: base={1-5, 11-15}, novel={6-10}
  Fold 2: base={1-10}, novel={11-15}

输出目录结构 | Output::

    data/iSAID-5i/sam_rsp/
    ├── lists/
    │   ├── fold0/
    │   │   ├── base_train.txt       (base class tiles: image mask pairs)
    │   │   ├── base_train_classes.txt (dict: class_id → [(img, mask), ...])
    │   │   ├── novel_train.txt
    │   │   ├── novel_train_classes.txt
    │   │   ├── novel_val.txt
    │   │   └── novel_val_classes.txt
    │   ├── fold1/ ...
    │   └── fold2/ ...
    └── all_tiles.txt  (all tiles → semantic mask, class info)

用法 | Usage::

    python tools/sam_rsp_prepare_isaid.py --data-root data/iSAID-5i
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adasam.datasets.isaid_5i import ISAID5I_FOLDS

CLASS_NAMES = [
    "BG", "ship", "storage_tank", "baseball_diamond", "tennis_court",
    "basketball_court", "ground_track_field", "bridge", "large_vehicle",
    "small_vehicle", "helicopter", "swimming_pool", "roundabout",
    "soccer_ball_field", "plane", "harbor",
]


def scan_tiles(img_dir: Path, ann_dir: Path) -> dict[str, dict]:
    """扫描目录下所有 tile, 读取 mask 确定类别.

    Scan all tiles in a directory, read their masks to determine classes.
    Returns: {tile_id: {"img_path": str, "ann_path": str, "classes": [int]}}
    """
    from collections import defaultdict as dd

    tiles: dict[str, dict] = {}
    img_files = sorted(img_dir.glob("*.png"))
    print(f"  Scanning {len(img_files)} tiles in {img_dir.name}...")

    for img_path in tqdm(img_files, desc=f"  {img_dir.name}", leave=False):
        tile_id = img_path.stem  # e.g. "P0000_1648_1904_3090_3346"
        ann_path = ann_dir / f"{tile_id}_instance_color_RGB.png"
        if not ann_path.exists():
            continue

        ann = cv2.imread(str(ann_path), cv2.IMREAD_GRAYSCALE)
        if ann is None:
            continue

        classes = sorted({int(c) for c in np.unique(ann) if 1 <= c <= 15})
        if not classes:
            continue

        tiles[tile_id] = {
            "img_path": str(img_path.absolute()),
            "ann_path": str(ann_path.absolute()),
            "classes": classes,
        }

    return tiles


def prepare(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root) / "iSAID"

    # Scan all tiles from both train and val directories
    all_tiles: dict[str, dict] = {}
    for split in ("train", "val"):
        img_dir = data_root / split / "images"
        ann_dir = data_root / split / "semantic_png"
        if img_dir.exists():
            tiles = scan_tiles(img_dir, ann_dir)
            for tid, info in tiles.items():
                info["split"] = split
            all_tiles.update(tiles)

    print(f"\nTotal tiles scanned: {len(all_tiles)}")

    # Count class distribution
    from collections import Counter
    class_counts = Counter()
    for info in all_tiles.values():
        for c in info["classes"]:
            class_counts[c] += 1
    print("Class distribution across ALL tiles:")
    for c in range(1, 16):
        print(f"  class {c:>2d} ({CLASS_NAMES[c]:<20s}): {class_counts[c]:>5d} tiles")

    # Output directory
    out_root = Path(args.data_root) / "sam_rsp"
    out_root.mkdir(parents=True, exist_ok=True)

    for fold in range(3):
        fold_def = ISAID5I_FOLDS[fold]
        base_classes = set(fold_def["train"])   # [6-15] for fold 0
        novel_classes = set(fold_def["test"])    # [1-5] for fold 0

        print(f"\n{'='*60}")
        print(f"  Fold {fold}: base={sorted(base_classes)}, novel={sorted(novel_classes)}")
        print(f"{'='*60}")

        # Build class→tile mappings for this fold
        base_train: dict[int, list[tuple[str, str]]] = defaultdict(list)
        novel_train: dict[int, list[tuple[str, str]]] = defaultdict(list)
        novel_val: dict[int, list[tuple[str, str]]] = defaultdict(list)

        for tile_id, info in all_tiles.items():
            img_p, ann_p = info["img_path"], info["ann_path"]
            item = (img_p, ann_p)
            split = info["split"]

            for cls_id in info["classes"]:
                if cls_id in base_classes:
                    base_train[cls_id].append(item)
                elif cls_id in novel_classes:
                    if split == "train":
                        novel_train[cls_id].append(item)
                    else:
                        novel_val[cls_id].append(item)

        # Write output for this fold
        fold_dir = out_root / "lists" / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        def write_lists(name: str, cls_dict: dict[int, list], desc: str) -> None:
            """Write data_list and sub_class_file_list files."""
            # Deduplicated data_list (each tile appears once even if multiple classes)
            seen: set[str] = set()
            data_list: list[tuple[str, str]] = []
            for items in cls_dict.values():
                for img_p, ann_p in items:
                    key = img_p
                    if key not in seen:
                        seen.add(key)
                        data_list.append((img_p, ann_p))

            # Write data_list
            dl_path = fold_dir / f"{name}.txt"
            with open(dl_path, "w", encoding="utf-8") as f:
                for img_p, ann_p in data_list:
                    f.write(f"{img_p} {ann_p}\n")

            # Write class→file mapping
            cl_path = fold_dir / f"{name}_classes.txt"
            with open(cl_path, "w", encoding="utf-8") as f:
                serializable = {str(k): v for k, v in cls_dict.items()}
                f.write(str(serializable))

            n_tiles = len(data_list)
            n_pairs = sum(len(v) for v in cls_dict.values())
            print(f"  {desc:30s}: {n_tiles:>5d} tiles, {n_pairs:>5d} class-tile pairs")
            for cls_id in sorted(cls_dict.keys()):
                n = len(cls_dict[cls_id])
                print(f"    class {cls_id:>2d} ({CLASS_NAMES[cls_id]:<20s}): {n:>5d} tiles")

        write_lists("base_train", base_train, "Base (train)")
        write_lists("novel_train", novel_train, "Novel (train)")
        write_lists("novel_val", novel_val, "Novel (val)")

    print(f"\n[prepare] Done. Output: {out_root}")


def main() -> None:
    p = argparse.ArgumentParser(description="SAM-RSP iSAID-5i Data Preparation")
    p.add_argument("--data-root", type=str, default=str(_REPO_ROOT / "data" / "iSAID-5i"),
                   help="iSAID-5i data root directory")
    args = p.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
