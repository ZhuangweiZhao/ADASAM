"""
iSAID-5i 数据集统计分析 | iSAID-5i Semantic Segmentation Dataset Analysis.
=========================================================================

对 iSAID-5i 语义分割数据集做统计分析:
  ① 类别分布 (每类像素数 / tile 数 / 前景覆盖率)
  ② 类共现 (tile 内多类共存热图)
  ③ Fold 划分概览 (3 折 base/novel 分布)

Reads PNG masks directly; no COCO JSON dependency.

输出 | Output:
    - 终端汇总表 | console summary tables
    - <output>/report.json

用法 | Usage::

    python tools/analysis/analyze_dataset.py
    python tools/analysis/analyze_dataset.py --data-root data/iSAID-5i --plots
    python tools/analysis/analyze_dataset.py --fold 0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── iSAID-5i constants ──
NUM_CLASSES = 15
CLASS_NAMES = {
    1: "ship", 2: "storage_tank", 3: "baseball_diamond",
    4: "tennis_court", 5: "basketball_court", 6: "ground_track_field",
    7: "bridge", 8: "large_vehicle", 9: "small_vehicle",
    10: "helicopter", 11: "swimming_pool", 12: "roundabout",
    13: "soccer_ball_field", 14: "plane", 15: "harbor",
}

ISAID5I_FOLDS = {
    0: {"novel": [1, 2, 3, 4, 5],      "base": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]},
    1: {"novel": [6, 7, 8, 9, 10],      "base": [1, 2, 3, 4, 5, 11, 12, 13, 14, 15]},
    2: {"novel": [11, 12, 13, 14, 15],  "base": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
}

BG_CLASS = 0
TILE_SIZE = 256


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _scan_tiles(img_dir: Path, mask_dir: Path) -> list[tuple[str, Path, Path]]:
    """Scan for (tile_id, image_path, mask_path) triplets.

    Image:  {img_dir}/{tile_id}.png
    Mask:   {mask_dir}/{tile_id}_instance_color_RGB.png
    """
    tiles = []
    for img_path in sorted(img_dir.glob("*.png")):
        tile_id = img_path.stem
        mask_path = mask_dir / f"{tile_id}_instance_color_RGB.png"
        if mask_path.exists():
            tiles.append((tile_id, img_path, mask_path))
    return tiles


def _analyze_mask(mask: np.ndarray) -> dict:
    """Analyze a single semantic mask.

    Returns per-class pixel counts and presence flags.
    """
    result: dict[int, int] = defaultdict(int)
    unique, counts = np.unique(mask, return_counts=True)
    for cls_id, cnt in zip(unique, counts):
        cls_id = int(cls_id)
        if 1 <= cls_id <= NUM_CLASSES:
            result[cls_id] = int(cnt)
    return dict(result)


# ═══════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════

def analyze(data_root: Path, splits: list[str]) -> dict:
    """Run full analysis on iSAID-5i dataset."""

    report: dict = {"splits": {}, "folds": {}}
    total_pixels = TILE_SIZE * TILE_SIZE

    for split in splits:
        img_dir = data_root / "iSAID" / split / "images"
        mask_dir = data_root / "iSAID" / split / "semantic_png"
        if not img_dir.is_dir():
            print(f"  [skip] images dir not found: {img_dir}")
            continue
        if not mask_dir.is_dir():
            print(f"  [skip] masks dir not found: {mask_dir}")
            continue

        tiles = _scan_tiles(img_dir, mask_dir)
        n_tiles = len(tiles)
        if n_tiles == 0:
            print(f"  [skip] no tiles in {img_dir}")
            continue

        print(f"  {split}: scanning {n_tiles} tiles...")

        # Per-class accumulators
        cls_pixels: dict[int, int] = defaultdict(int)      # total pixels per class
        cls_tiles: dict[int, set[str]] = defaultdict(set)   # tile names per class
        cls_fg_ratios: dict[int, list[float]] = defaultdict(list)  # FG coverage per tile
        cooc = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)  # co-occurrence
        all_present: list[set[int]] = []  # per-tile class sets

        for tile_id, img_path, mask_path in tiles:
            mask = _read_mask_palette(mask_path)
            if mask is None:
                continue

            pixel_stats = _analyze_mask(mask)

            present = set(pixel_stats.keys())
            all_present.append(present)

            for cls_id, px in pixel_stats.items():
                cls_pixels[cls_id] += px
                cls_tiles[cls_id].add(tile_id)
                cls_fg_ratios[cls_id].append(px / total_pixels)

            # Co-occurrence: each pair of classes present in this tile
            p_list = sorted(present)
            for i, ci in enumerate(p_list):
                for cj in p_list[i:]:
                    cooc[ci - 1, cj - 1] += 1
                    if ci != cj:
                        cooc[cj - 1, ci - 1] += 1

        # Build per-class summary
        per_class = {}
        for cls_id in range(1, NUM_CLASSES + 1):
            px = cls_pixels.get(cls_id, 0)
            n_tiles_cls = len(cls_tiles.get(cls_id, set()))
            ratios = cls_fg_ratios.get(cls_id, [])
            per_class[str(cls_id)] = {
                "name": CLASS_NAMES[cls_id],
                "total_pixels": px,
                "total_pixels_pct": px / max(n_tiles * total_pixels, 1) * 100,
                "n_tiles": n_tiles_cls,
                "tile_coverage_pct": n_tiles_cls / max(n_tiles, 1) * 100,
                "fg_ratio_mean": float(np.mean(ratios)) if ratios else 0.0,
                "fg_ratio_median": float(np.median(ratios)) if ratios else 0.0,
                "fg_ratio_p95": float(np.percentile(ratios, 95)) if ratios else 0.0,
            }

        # Tile-level: class count distribution
        n_classes_per_tile = [len(s) for s in all_present]
        tile_cls_counts = np.asarray(n_classes_per_tile, dtype=np.int64)

        report["splits"][split] = {
            "n_tiles": n_tiles,
            "per_class": per_class,
            "cooccurrence": cooc.tolist(),
            "tile_stats": {
                "classes_per_tile_mean": float(tile_cls_counts.mean()) if tile_cls_counts.size else 0.0,
                "classes_per_tile_median": float(np.median(tile_cls_counts)) if tile_cls_counts.size else 0.0,
                "classes_per_tile_max": int(tile_cls_counts.max()) if tile_cls_counts.size else 0,
                "single_class_tiles": int((tile_cls_counts == 1).sum()),
                "empty_tiles": int((tile_cls_counts == 0).sum()),
            },
        }

    # Fold analysis
    for fold in (0, 1, 2):
        fd = ISAID5I_FOLDS[fold]
        entry: dict = {
            "base": [f"{c}({CLASS_NAMES[c]})" for c in fd["base"]],
            "novel": [f"{c}({CLASS_NAMES[c]})" for c in fd["novel"]],
        }
        for split_name, split_data in report["splits"].items():
            pc = split_data["per_class"]
            entry[f"{split_name}_base_pixels"] = sum(
                pc.get(str(c), {}).get("total_pixels", 0) for c in fd["base"]
            )
            entry[f"{split_name}_novel_pixels"] = sum(
                pc.get(str(c), {}).get("total_pixels", 0) for c in fd["novel"]
            )
            entry[f"{split_name}_base_tiles"] = sum(
                pc.get(str(c), {}).get("n_tiles", 0) for c in fd["base"]
            )
            entry[f"{split_name}_novel_tiles"] = sum(
                pc.get(str(c), {}).get("n_tiles", 0) for c in fd["novel"]
            )
        report["folds"][str(fold)] = entry

    return report


def _read_mask_palette(path: Path) -> np.ndarray | None:
    """Read palette PNG mask (IMREAD_UNCHANGED gives class IDs directly)."""
    try:
        import cv2
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            return None
        return mask.astype(np.int64)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════
# Printing
# ═══════════════════════════════════════════════════════════════════

def _print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
        for i, h in enumerate(headers)
    ]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(f"\n── {title} ──")
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(v.ljust(w) for v, w in zip(r, widths)))


def print_report(report: dict) -> None:
    for split_name, split_data in report["splits"].items():
        n_tiles = split_data["n_tiles"]
        ts = split_data["tile_stats"]
        print(f"\n{'=' * 78}")
        print(f"  SPLIT: {split_name}  ({n_tiles} tiles)")
        print(f"  classes/tile: mean={ts['classes_per_tile_mean']:.1f} "
              f"median={ts['classes_per_tile_median']:.0f} "
              f"max={ts['classes_per_tile_max']}  "
              f"single_class={ts['single_class_tiles']}  "
              f"empty={ts['empty_tiles']}")
        print(f"{'=' * 78}")

        # ① Class distribution
        pc = split_data["per_class"]
        rows = []
        for c in sorted(pc, key=lambda x: pc[x]["total_pixels"], reverse=True):
            r = pc[c]
            rows.append([
                c,
                r["name"],
                f"{r['total_pixels']:>12,}",
                f"{r['total_pixels_pct']:.2f}%",
                f"{r['n_tiles']:>5}",
                f"{r['tile_coverage_pct']:.1f}%",
                f"{r['fg_ratio_mean']:.3f}",
                f"{r['fg_ratio_median']:.3f}",
                f"{r['fg_ratio_p95']:.3f}",
            ])
        _print_table(
            "① 类别分布 | class distribution",
            ["id", "class", "pixels", "px%", "tiles", "tile%", "fg_mean", "fg_med", "fg_p95"],
            rows,
        )

        # ② Co-occurrence top pairs
        cooc = np.asarray(split_data["cooccurrence"])
        pairs = [
            (int(cooc[i, j]), i + 1, j + 1)
            for i in range(NUM_CLASSES)
            for j in range(i + 1, NUM_CLASSES)
            if cooc[i, j] > 0
        ]
        pairs.sort(reverse=True)
        print(f"\n  top co-occurring class pairs:")
        for n, ci, cj in pairs[:10]:
            print(f"    {CLASS_NAMES[ci]} + {CLASS_NAMES[cj]}: {n:,} tiles")

    # ③ Fold summary
    folds = report.get("folds", {})
    if folds:
        print(f"\n{'=' * 78}")
        print(f"  ③ FOLD SPLITS (base / novel)")
        print(f"{'=' * 78}")
        for fold, entry in folds.items():
            print(f"\n  fold {fold}:")
            print(f"    base  = [{', '.join(entry['base'])}]")
            print(f"    novel = [{', '.join(entry['novel'])}]")
            for split_name in report["splits"]:
                bp = entry.get(f"{split_name}_base_pixels", 0)
                np_px = entry.get(f"{split_name}_novel_pixels", 0)
                bt = entry.get(f"{split_name}_base_tiles", 0)
                nt = entry.get(f"{split_name}_novel_tiles", 0)
                total_px = bp + np_px
                print(f"    {split_name}: base={bp:,}px ({bp/max(total_px,1)*100:.0f}%)  "
                      f"novel={np_px:,}px ({np_px/max(total_px,1)*100:.0f}%)  "
                      f"|  base_tiles={bt:,}  novel_tiles={nt:,}")


# ═══════════════════════════════════════════════════════════════════
# Optional plots
# ═══════════════════════════════════════════════════════════════════

def make_plots(report: dict, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze] matplotlib missing; skip plots (pip install matplotlib)")
        return

    names = [CLASS_NAMES[c] for c in range(1, NUM_CLASSES + 1)]
    idx = np.arange(NUM_CLASSES)

    for split_name, split_data in report["splits"].items():
        pc = split_data["per_class"]
        pxs = [pc.get(str(c), {}).get("total_pixels", 0) for c in range(1, NUM_CLASSES + 1)]
        tiles_cls = [pc.get(str(c), {}).get("n_tiles", 0) for c in range(1, NUM_CLASSES + 1)]

        # Class distribution (pixels)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.bar(idx, pxs, color="steelblue")
        ax1.set_yscale("log")
        ax1.set_xticks(idx)
        ax1.set_xticklabels(names, rotation=45, ha="right")
        ax1.set_ylabel("pixels (log)")
        ax1.set_title(f"Pixels per class ({split_name})")

        ax2.bar(idx, tiles_cls, color="darkorange")
        ax2.set_xticks(idx)
        ax2.set_xticklabels(names, rotation=45, ha="right")
        ax2.set_ylabel("tiles")
        ax2.set_title(f"Tiles per class ({split_name})")
        fig.tight_layout()
        fig.savefig(out_dir / f"class_distribution_{split_name}.png", dpi=150)
        plt.close(fig)

        # FG ratio boxplot
        ratios = [pc.get(str(c), {}).get("fg_ratio_median", 0) for c in range(1, NUM_CLASSES + 1)]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(idx, ratios, color="forestgreen")
        ax.set_xticks(idx)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("median FG ratio")
        ax.set_title(f"Median foreground coverage per tile ({split_name})")
        fig.tight_layout()
        fig.savefig(out_dir / f"fg_coverage_{split_name}.png", dpi=150)
        plt.close(fig)

        # Co-occurrence heatmap
        cooc = np.asarray(split_data["cooccurrence"], dtype=np.float64)
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(np.log1p(cooc), cmap="YlOrRd")
        ax.set_xticks(idx)
        ax.set_xticklabels(names, rotation=90)
        ax.set_yticks(idx)
        ax.set_yticklabels(names)
        ax.set_title(f"Class co-occurrence, log(1+tiles) ({split_name})")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(out_dir / f"cooccurrence_{split_name}.png", dpi=150)
        plt.close(fig)

    print(f"[analyze] plots saved → {out_dir}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="iSAID-5i semantic segmentation dataset analyzer")
    p.add_argument("--data-root", default=str(_REPO_ROOT / "data" / "iSAID-5i"))
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--fold", type=int, default=None, help="show single fold detail (0/1/2)")
    p.add_argument("--plots", action="store_true", help="save matplotlib PNGs")
    p.add_argument("--output-dir", default=str(_REPO_ROOT / "runs" / "dataset_analysis"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analyze] data_root = {data_root}")
    print(f"[analyze] splits = {args.splits}")
    print()

    report = analyze(data_root, args.splits)
    report["meta"] = {
        "data_root": str(data_root),
        "tile_size": TILE_SIZE,
        "splits": args.splits,
        "num_classes": NUM_CLASSES,
    }

    print_report(report)

    if args.fold is not None:
        fd = ISAID5I_FOLDS[args.fold]
        print(f"\n── fold {args.fold} detail ──")
        print(f"  base  ({len(fd['base'])}): {[CLASS_NAMES[c] for c in fd['base']]}")
        print(f"  novel ({len(fd['novel'])}): {[CLASS_NAMES[c] for c in fd['novel']]}")

    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[analyze] report → {report_path}")

    if args.plots:
        make_plots(report, out_dir)


if __name__ == "__main__":
    main()
