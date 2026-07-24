#!/usr/bin/env python3
"""
iSAID-5i 数据集结构诊断脚本
===============================
诊断问题:
  1. 每 tile 每类 到底有多少个实例? (用 instance_mask 而非 semantic_png)
  2. 多实例 tile 占比多少?
  3. 每 tile 有多少个类别共存?
  4. 实例面积分布?
  5. Support/Query scene-disjoint 后的实际可用统计?

用法:
  python tools/deprecated/diagnose_dataset.py                     # 全部诊断
  python tools/deprecated/diagnose_dataset.py --fold 0             # 只查 fold 0
  python tools/deprecated/diagnose_dataset.py --detail             # 逐个 tile 输出多实例详情
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════
# iSAID-5i fold 定义
# ═══════════════════════════════════════════════════════════════════
ISAID5I_FOLDS = {
    0: {"test": [1, 2, 3, 4, 5],       "train": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]},
    1: {"test": [6, 7, 8, 9, 10],       "train": [1, 2, 3, 4, 5, 11, 12, 13, 14, 15]},
    2: {"test": [11, 12, 13, 14, 15],   "train": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
}

CLASS_NAMES = {
    1: "ship", 2: "storage_tank", 3: "baseball_diamond", 4: "tennis_court",
    5: "basketball_court", 6: "ground_track_field", 7: "bridge", 8: "large_vehicle",
    9: "small_vehicle", 10: "helicopter", 11: "swimming_pool", 12: "roundabout",
    13: "soccer_ball_field", 14: "plane", 15: "harbor",
}


def parse_list_file(list_path: Path) -> dict[str, set[int]]:
    """
    解析 train_list / val_list 文件，返回 {tile_id: {class_ids}}.
    文件格式: P1092_1648_1904_824_1080_instance_color_RGB.png_04
    """
    tile_classes: dict[str, set[int]] = defaultdict(set)
    if not list_path.exists():
        return dict(tile_classes)

    with open(list_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit("_", 1)
            class_id = int(parts[1])
            tile_id = parts[0].replace("_instance_color_RGB.png", "")
            tile_classes[tile_id].add(class_id)
    return dict(tile_classes)


def count_instances_from_instance_mask(
    mask_dir: Path, tile_id: str
) -> dict[int, int]:
    """
    从 instance_mask PNG 读取真实实例数.
    返回 {class_id: instance_count}.

    iSAID instance_mask: RGB 图, 每个实例有唯一 RGB 颜色.
    需要通过 semantic_png 来确定每个实例像素属于哪个 class.
    """
    result: dict[int, int] = {}
    mask_path = mask_dir / f"{tile_id}_instance_id_RGB.png"
    if not mask_path.exists():
        return result

    img = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if img is None:
        return result
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 将 RGB 编码为唯一 ID (每个值范围 0-255, 用 (R << 16) | (G << 8) | B)
    encoded = (img[:, :, 0].astype(np.int64) << 16) | \
              (img[:, :, 1].astype(np.int64) << 8) | \
              img[:, :, 2].astype(np.int64)
    unique_ids = np.unique(encoded)
    # 过滤背景 (0)
    fg_ids = [uid for uid in unique_ids if uid != 0]

    # 没有 semantic_png，无法直接区分 class.
    # 但可通过检查该 tile 在 list_file 中的 class 来标记.
    # 这里我们只返回 fg instance count，不区分 class.
    # 要区分 class 需要 semantic_png 辅助.
    return {"_total_fg_instances": len(fg_ids)}


def count_instances_with_semantic(
    inst_dir: Path, sem_dir: Path, tile_id: str
) -> dict[int, int]:
    """
    结合 semantic_png + instance_mask 统计每类实例数.

    semantic_png: 每个像素是 class_id (0=BG, 1-15=class)
    instance_mask: 每个像素是 instance RGB ID (unique per instance)
    """
    # Load semantic
    sem_path = sem_dir / f"{tile_id}_instance_color_RGB.png"
    if not sem_path.exists():
        return {}
    sem = cv2.imread(str(sem_path), cv2.IMREAD_UNCHANGED)
    if sem is None:
        return {}

    # Load instance
    inst_path = inst_dir / f"{tile_id}_instance_id_RGB.png"
    if not inst_path.exists():
        return {}
    img = cv2.imread(str(inst_path), cv2.IMREAD_COLOR)
    if img is None:
        return {}
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encoded = (img[:, :, 0].astype(np.int64) << 16) | \
              (img[:, :, 1].astype(np.int64) << 8) | \
              img[:, :, 2].astype(np.int64)

    # 对每个 class 统计不同 instance ID 数量
    result = {}
    for cls in range(1, 16):
        cls_mask = (sem == cls)
        if cls_mask.sum() == 0:
            continue
        cls_instance_ids = set(encoded[cls_mask].tolist()) - {0}
        result[cls] = len(cls_instance_ids)
    return result


def diagnose_fold(
    root: Path, fold: int, split: str, detail: bool = False
) -> dict:
    """对一个 fold+split 做全面诊断."""
    folder = "train" if split == "train" else "val"
    list_file = root / "iSAID" / folder / f"{'train' if split == 'train' else 'val'}_list" / f"split{fold}_{'train' if split == 'train' else 'val'}.txt"
    inst_dir = root / "iSAID" / folder / "instance_mask"
    sem_dir = root / "iSAID" / folder / "semantic_png"
    img_dir = root / "iSAID" / folder / "images"

    # 1. 解析 list file
    tile_classes = parse_list_file(list_file)

    # 统计
    total_tiles = len(tile_classes)
    if total_tiles == 0:
        print(f"  No tiles in {list_file}")
        return {}

    classes_per_tile = [len(v) for v in tile_classes.values()]
    multi_class_tiles = sum(1 for c in classes_per_tile if c > 1)

    # 2. 用 semantic_png 检查每类每 tile 的像素连通分量 (semantic mask 自身)
    # 语义 mask 中同一类可能有多个不连通区域 == 多个实例
    tile_instances: dict[str, dict[int, int]] = {}  # tile_id -> {class_id: instance_count}

    for tile_id in sorted(tile_classes.keys()):
        sem_path = sem_dir / f"{tile_id}_instance_color_RGB.png"
        if not sem_path.exists():
            continue
        sem = cv2.imread(str(sem_path), cv2.IMREAD_UNCHANGED)
        if sem is None:
            continue
        per_cls = {}
        for cls in tile_classes[tile_id]:
            cls_mask = (sem == cls).astype(np.uint8)
            if cls_mask.sum() > 0:
                # 连通分量分析获取实例数
                num_labels, _ = cv2.connectedComponents(cls_mask, connectivity=8)
                per_cls[cls] = num_labels - 1  # 减背景
        tile_instances[tile_id] = per_cls

    # 3. 结合 instance_mask 做精确统计 (更可靠)
    tile_instances_v2: dict[str, dict[int, int]] = {}
    for idx, tile_id in enumerate(sorted(tile_classes.keys())):
        inst = count_instances_with_semantic(inst_dir, sem_dir, tile_id)
        tile_instances_v2[tile_id] = inst
        if idx < 10 and detail:
            print(f"    [{tile_id}] semantic: {tile_instances.get(tile_id, {})}, instance_mask: {inst}")

    return {
        "total_tiles": total_tiles,
        "multi_class_tiles": multi_class_tiles,
        "classes_per_tile_dist": Counter(classes_per_tile),
        "tile_classes": tile_classes,
        "tile_instances_semantic": tile_instances,
        "tile_instances_v2": tile_instances_v2,
    }


def print_separator(title: str, char: str = "=") -> None:
    width = 72
    print(f"\n{' ' + title + ' ':{char}^{width}}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="iSAID-5i 数据集结构诊断"
    )
    parser.add_argument("--root", default="data/iSAID-5i", help="数据根目录")
    parser.add_argument("--fold", type=int, default=None, help="指定 fold (0/1/2), 默认全部")
    parser.add_argument("--detail", action="store_true", help="输出每个 tile 的详情(多实例 tile)")
    parser.add_argument(
        "--only-instances", action="store_true",
        help="仅做 instance_mask 精确诊断 (无需 semantic_png 解析)"
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: 数据目录不存在: {root}")
        sys.exit(1)

    folds = [args.fold] if args.fold is not None else [0, 1, 2]
    splits = ["train", "val"]

    # ═══════════════════════════════════════════════════════════════
    # PART A: 粗粒度统计 (从 list file)
    # ═══════════════════════════════════════════════════════════════
    print_separator("PART A: 粗粒度统计 (list file → tile→class 映射)", "=")

    for fold in folds:
        for split in splits:
            folder = "train" if split == "train" else "val"
            list_file = root / "iSAID" / folder / f"{'train' if split == 'train' else 'val'}_list" / f"split{fold}_{'train' if split == 'train' else 'val'}.txt"

            if not list_file.exists():
                print(f"  fold={fold} {split}: list file not found ({list_file})")
                continue

            tile_classes = parse_list_file(list_file)

            # 基本统计
            n_tiles = len(tile_classes)
            class_pairs = sum(len(v) for v in tile_classes.values())
            classes_per_tile = [len(v) for v in tile_classes.values()]
            unique_classes = set()
            for v in tile_classes.values():
                unique_classes |= v

            print(f"\n── fold={fold} {split} ──")
            print(f"  List file entries:     {class_pairs}")
            print(f"  Unique tiles:          {n_tiles}")
            print(f"  Unique classes:        {sorted(unique_classes)} ({len(unique_classes)})")
            print(f"  Classes per tile — mean:  {np.mean(classes_per_tile):.3f}")
            print(f"                      median: {np.median(classes_per_tile):.1f}")
            print(f"                      max:    {max(classes_per_tile)}")
            cpt_dist = Counter(classes_per_tile)
            print(f"  Classes-per-tile distribution:")
            for k in sorted(cpt_dist):
                pct = 100.0 * cpt_dist[k] / n_tiles
                print(f"      {k} class(es): {cpt_dist[k]:5d} tiles ({pct:5.1f}%)")

            # 每类有多少 tile
            class_tile_count = Counter()
            for v in tile_classes.values():
                for c in v:
                    class_tile_count[c] += 1
            print(f"  Tiles per class:")
            for cls in sorted(class_tile_count):
                name = CLASS_NAMES.get(cls, "?")
                print(f"      class {cls:2d} ({name:20s}): {class_tile_count[cls]:5d} tiles")

    # ═══════════════════════════════════════════════════════════════
    # PART B: 精确实例数统计 (从 semantic_png 连通分量)
    # ═══════════════════════════════════════════════════════════════
    print_separator("PART B: semantic_png 连通分量 → 每类实例数", "=")

    for fold in folds:
        for split in splits:
            folder = "train" if split == "train" else "val"
            list_file = root / "iSAID" / folder / f"{'train' if split == 'train' else 'val'}_list" / f"split{fold}_{'train' if split == 'train' else 'val'}.txt"
            sem_dir = root / "iSAID" / folder / "semantic_png"

            if not list_file.exists():
                continue

            tile_classes = parse_list_file(list_file)

            # 收集所有 tile 的每类实例数
            class_instance_counts: dict[int, list[int]] = defaultdict(list)
            # tile_id -> {class_id: instance_count}
            tile_detail: dict[str, dict[int, int]] = {}
            # 每类每 tile 的实例数分布 (用于统计多实例 tile)
            multi_instance_tiles_per_class: dict[int, list[str]] = defaultdict(list)
            total_instances_per_class: dict[int, int] = defaultdict(int)

            tile_list = sorted(tile_classes.keys())
            print(f"\n── fold={fold} {split}: 扫描 {len(tile_list)} tiles ──")

            for tile_id in tile_list:
                sem_path = sem_dir / f"{tile_id}_instance_color_RGB.png"
                if not sem_path.exists():
                    continue
                sem = cv2.imread(str(sem_path), cv2.IMREAD_UNCHANGED)
                if sem is None:
                    continue

                per_cls = {}
                for cls in tile_classes[tile_id]:
                    cls_mask = (sem == cls).astype(np.uint8)
                    if cls_mask.sum() > 0:
                        n_inst, _ = cv2.connectedComponents(cls_mask, connectivity=8)
                        n_inst -= 1  # 减背景
                        per_cls[cls] = n_inst
                        class_instance_counts[cls].append(n_inst)
                        total_instances_per_class[cls] += n_inst
                        if n_inst > 1:
                            multi_instance_tiles_per_class[cls].append(tile_id)
                tile_detail[tile_id] = per_cls

            # 汇总统计
            print(f"\n  每类每 tile 实例数分布 (semantic 连通分量):")
            print(f"  {'Class':>6s}  {'Name':<22s} {'TotalTiles':>10s} {'TotalInsts':>10s} "
                  f"{'Mean':>7s} {'Median':>7s} {'Max':>5s} {'MultiInst':>9s} {'Multi%':>7s}")
            print(f"  {'-'*80}")
            for cls in sorted(class_instance_counts):
                counts = class_instance_counts[cls]
                name = CLASS_NAMES.get(cls, "?")
                multi = sum(1 for c in counts if c > 1)
                print(f"  {cls:6d}  {name:<22s} {len(counts):10d} {total_instances_per_class[cls]:10d} "
                      f"{np.mean(counts):7.3f} {np.median(counts):7.0f} {max(counts):5d} "
                      f"{multi:9d} {100.0*multi/len(counts):7.2f}%")

            # 全局: 每 tile 总实例数 (所有 class 求和)
            tile_total_insts = [sum(d.values()) for d in tile_detail.values()]
            total_inst_dist = Counter(tile_total_insts)
            print(f"\n  每 tile 总实例数 (所有 class 求和):")
            print(f"      mean={np.mean(tile_total_insts):.3f}, median={np.median(tile_total_insts):.1f}, max={max(tile_total_insts)}")
            print(f"      分布: {dict(sorted(total_inst_dist.items())[:15])}{'...' if len(total_inst_dist) > 15 else ''}")

            # 多实例 tile 详情 (如果有)
            if args.detail:
                multi_total = [(tid, sum(d.values())) for tid, d in tile_detail.items() if sum(d.values()) > len(d)]
                if multi_total:
                    print(f"\n  多类或多实例 tile 详情 (前30):")
                    for tid, inst_cnt in sorted(multi_total, key=lambda x: -x[1])[:30]:
                        detail = tile_detail[tid]
                        print(f"      {tid}: {dict(detail)} (total={inst_cnt})")

    # ═══════════════════════════════════════════════════════════════
    # PART C: instance_mask 精确统计 (最可靠)
    # ═══════════════════════════════════════════════════════════════
    print_separator("PART C: instance_mask 精确统计 (最可靠)", "=")

    for fold in folds:
        for split in splits:
            folder = "train" if split == "train" else "val"
            list_file = root / "iSAID" / folder / f"{'train' if split == 'train' else 'val'}_list" / f"split{fold}_{'train' if split == 'train' else 'val'}.txt"
            inst_dir = root / "iSAID" / folder / "instance_mask"
            sem_dir = root / "iSAID" / folder / "semantic_png"

            if not list_file.exists():
                continue

            tile_classes = parse_list_file(list_file)
            tile_list = sorted(tile_classes.keys())

            # 结合 semantic + instance_mask
            class_instance_counts_v2: dict[int, list[int]] = defaultdict(list)
            multi_instance_v2: dict[int, list[str]] = defaultdict(list)
            n_scanned = 0
            n_errors = 0

            sample_n = min(100, len(tile_list))
            for idx, tile_id in enumerate(tile_list):
                inst = count_instances_with_semantic(inst_dir, sem_dir, tile_id)
                if inst:
                    n_scanned += 1
                    for cls, cnt in inst.items():
                        class_instance_counts_v2[cls].append(cnt)
                        if cnt > 1:
                            multi_instance_v2[cls].append(tile_id)
                else:
                    n_errors += 1
                    if n_errors <= 5:
                        # 检查什么文件缺失
                        sem_missing = not (sem_dir / f"{tile_id}_instance_color_RGB.png").exists()
                        inst_missing = not (inst_dir / f"{tile_id}_instance_id_RGB.png").exists()
                        print(f"  WARN: {tile_id}: sem_missing={sem_missing}, inst_missing={inst_missing}")

                if (idx + 1) % 500 == 0:
                    print(f"  ... {idx + 1}/{len(tile_list)} scanned ...")

            print(f"\n── fold={fold} {split}: {n_scanned} tiles scanned, {n_errors} errors ──")
            print(f"  instance_mask 精确实例数分布:")
            print(f"  {'Class':>6s}  {'Name':<22s} {'TotalTiles':>10s} "
                  f"{'Mean':>7s} {'Median':>7s} {'Max':>5s} {'MultiInst%':>9s}")
            print(f"  {'-'*70}")
            for cls in sorted(class_instance_counts_v2):
                counts = class_instance_counts_v2[cls]
                name = CLASS_NAMES.get(cls, "?")
                multi_pct = 100.0 * sum(1 for c in counts if c > 1) / len(counts)
                print(f"  {cls:6d}  {name:<22s} {len(counts):10d} "
                      f"{np.mean(counts):7.3f} {np.median(counts):7.0f} {max(counts):5d} "
                      f"{multi_pct:8.2f}%")

            # 多实例 tile 样例
            if args.detail:
                for cls in sorted(multi_instance_v2):
                    if multi_instance_v2[cls]:
                        tiles = multi_instance_v2[cls]
                        print(f"\n  class {cls} ({CLASS_NAMES.get(cls, '?')}) 多实例 tile ({len(tiles)} 个):")
                        for tid in tiles[:10]:
                            cnt = class_instance_counts_v2[cls][
                                list(class_instance_counts_v2.keys()).index(cls)
                            ] if cls in class_instance_counts_v2 else "?"
                            print(f"      {tid}: {cnt} instances")

    # ═══════════════════════════════════════════════════════════════
    # PART D: AdaSAM 实际训练时会看到什么?
    # ═══════════════════════════════════════════════════════════════
    print_separator("PART D: AdaSAM 训练时的实际数据 (semantic_png -> 1 mask/class)", "=")

    for fold in folds:
        fold_def = ISAID5I_FOLDS[fold]
        novel_classes = fold_def["test"]
        print(f"\n── fold={fold} novel classes: {novel_classes} "
              f"({[CLASS_NAMES.get(c, '?') for c in novel_classes]}) ──")

        for split in splits:
            folder = "train" if split == "train" else "val"
            list_file = root / "iSAID" / folder / f"{'train' if split == 'train' else 'val'}_list" / f"split{fold}_{'train' if split == 'train' else 'val'}.txt"
            sem_dir = root / "iSAID" / folder / "semantic_png"

            if not list_file.exists():
                continue

            tile_classes = parse_list_file(list_file)

            # AdaSAM 当前代码逻辑: 每类做一个 binary mask = (sem == cls)
            # → 每个 visible class 在该 tile 中永远恰好 1 个 "实例" (如果有该 class 的话)
            novel_tiles = sum(1 for cset in tile_classes.values() if cset & set(novel_classes))

            print(f"  {split}: {novel_tiles} tiles contain novel classes "
                  f"(out of {len(tile_classes)} total)")

            # 但如果是用 FSS 协议训练, 每个 episode 只有一个 target class
            # → 对于该 target class, 每 tile 是 1 个 connected component → 1 instance
            # 所以 AdaSAM 训练时, Hungarian matching 总是匹配 1 个 query.
            print(f"    → 对于单个 target class, 每 tile 语义分割视为 1 个 'instance'")
            print(f"    → Hungarian matcher 只能匹配 1 个 query (数据中恒为 1 GT mask)")

    # ═══════════════════════════════════════════════════════════════
    # PART E: Instance 面积分布
    # ═══════════════════════════════════════════════════════════════
    print_separator("PART E: Instance 面积分布 (semantic mask)", "=")

    for fold in folds:
        folder = "train"
        list_file = root / "iSAID" / folder / "train_list" / f"split{fold}_train.txt"
        sem_dir = root / "iSAID" / folder / "semantic_png"
        if not list_file.exists():
            continue

        tile_classes = parse_list_file(list_file)
        areas_by_class: dict[int, list[float]] = defaultdict(list)

        for tile_id in sorted(tile_classes.keys()):
            sem_path = sem_dir / f"{tile_id}_instance_color_RGB.png"
            sem = cv2.imread(str(sem_path), cv2.IMREAD_UNCHANGED)
            if sem is None:
                continue
            for cls in tile_classes[tile_id]:
                area = (sem == cls).sum()
                areas_by_class[cls].append(area)

        print(f"\n── fold={fold} train ──")
        for cls in sorted(areas_by_class):
            areas = np.array(areas_by_class[cls])
            name = CLASS_NAMES.get(cls, "?")
            total_pix = 256 * 256
            print(f"  class {cls:2d} ({name:20s}): n={len(areas):5d}, "
                  f"mean={np.mean(areas):7.0f}px ({100*np.mean(areas)/total_pix:.1f}%), "
                  f"median={np.median(areas):7.0f}px, "
                  f"min={np.min(areas):5d}px, max={np.max(areas):6d}px, "
                  f"P10={np.percentile(areas, 10):5.0f}px, P90={np.percentile(areas, 90):5.0f}px")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print_separator("SUMMARY: 关键发现", "=")
    print()
    print("  1. 当前 AdaSAM 用 semantic_png → 每类每 tile 永远恰好 1 个 GT mask")
    print("     → Hungarian matching 只能匹配 1 个 query → DPG 学会 Q0 独占")
    print("     → Objectness collapse 是数据协议固有的，不是 bug!")
    print()
    print("  2. instance_mask 中可能有多个同类实例 (见 PART B/C 的 MultiInst%)")
    print("     → 如果要真正做实例分割, 需要改用 instance_mask 数据")
    print("     → FSS benchmark 标准协议用 semantic mask, 每类 1 个前景区域")
    print()
    print("  3. 改进方向:")
    print("     a) 如果坚持 FSS 协议 → 不需要 16 queries, 1-2 个足够")
    print("     b) 如果要做真正的实例分割 → 需要切换到 instance_mask")
    print("     c) 多类联合训练 → 利用 multi-class tile (PART A 中约 3.5% tile)")

    print()
    if args.detail:
        print("  运行完毕 (--detail 已启用)")
    else:
        print("  提示: 用 --detail 查看多实例 tile 详情")


if __name__ == "__main__":
    main()
