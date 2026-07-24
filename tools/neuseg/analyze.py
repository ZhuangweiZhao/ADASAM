"""
NEU-SEG 数据集分析 | Neu-SEG Dataset Analysis.
==============================================

对 NEU-SEG 二值建筑分割数据集进行多维度统计分析：
① 基础统计 (图像数/尺寸/划分)
② 实例 (连通域) 分析 (数量分布, 面积分布, 长宽比)
③ 前景覆盖率 (极端不平衡特征)
④ 逐图明细
⑤ Train/Val 分布对比

输出 | Output:
    - 终端汇总表
    - <output>/neu_seg_report.json
    - <output>/*.png  (--plots 时)

用法 | Usage::

    python tools/neuseg/analyze.py
    python tools/neuseg/analyze.py --data-root E:/.../Neu_seg --plots
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── 常量 | Constants ──
DEFAULT_DATA_ROOT = "E:/A_postgraduate_stude/AdaTile-FastSAM/data/Neu_seg"
IMAGE_H, IMAGE_W = 480, 640
TOTAL_PIXELS = IMAGE_H * IMAGE_W  # 307,200


# ═══════════════════════════════════════════════════════════════════
# 数据结构 | Data structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class InstanceStats:
    """单实例统计 | Per-instance statistics."""
    area: int           # 像素面积 | pixel area
    bbox_w: int         # 边界框宽 | bbox width
    bbox_h: int         # 边界框高 | bbox height
    aspect_ratio: float  # 长宽比 (max/min) | aspect ratio
    extent: float       # 填充率 (area / bbox_area) | extent


@dataclass
class ImageStats:
    """单图统计 | Per-image statistics."""
    name: str
    source: str          # "SDI" or "SPDI"
    split: str           # "train" or "val"
    n_instances: int
    fg_pixels: int
    fg_ratio: float      # 前景像素占比 | FG pixel ratio
    instances: list[InstanceStats] = field(default_factory=list)


@dataclass
class DatasetReport:
    """数据集分析报告 | Dataset analysis report."""
    total_images: int
    total_instances: int
    image_size: tuple[int, int]
    splits: dict[str, int]           # split → count
    sources: dict[str, int]          # source type → count
    # 实例级 | Instance-level
    instances_per_image: dict[str, float]   # mean / std / min / max
    area_stats: dict[str, float]            # mean / std / min / max / median
    fg_ratio_stats: dict[str, float]        # mean / std / min / max
    # 分布 | Distributions
    area_distribution: dict[str, int]       # bucket → count
    per_image: list[dict]                   # per-image breakdown


# ═══════════════════════════════════════════════════════════════════
# 分析逻辑 | Analysis logic
# ═══════════════════════════════════════════════════════════════════

def load_split_set(root: Path, split: str) -> set[str]:
    """读取 train.txt / val.txt 中的图像名集合."""
    path = root / f"{split}.txt"
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").strip().split("\n") if line.strip()}


def analyze_image(
    img_path: Path, ann_path: Path, name: str, split: str, source: str, min_area: int = 4
) -> Optional[ImageStats]:
    """分析单张图像 | Analyze a single image."""
    ann = cv2.imread(str(ann_path), cv2.IMREAD_GRAYSCALE)
    if ann is None:
        return None

    mask = (ann > 128).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    instances = []
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                      int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        instances.append(InstanceStats(
            area=area,
            bbox_w=w, bbox_h=h,
            aspect_ratio=max(w, h) / max(min(w, h), 1),
            extent=area / max(w * h, 1),
        ))

    fg_pixels = int(mask.sum())
    return ImageStats(
        name=name,
        source=source,
        split=split,
        n_instances=len(instances),
        fg_pixels=fg_pixels,
        fg_ratio=fg_pixels / TOTAL_PIXELS,
        instances=instances,
    )


def build_report(stats_list: list[ImageStats]) -> DatasetReport:
    """从图像统计列表构建汇总报告."""
    if not stats_list:
        raise ValueError("No images to analyze")

    total_instances = sum(s.n_instances for s in stats_list)
    splits = defaultdict(int)
    sources = defaultdict(int)
    for s in stats_list:
        splits[s.split] += 1
        sources[s.source] += 1

    # 实例数统计
    n_inst_arr = np.array([s.n_instances for s in stats_list])
    instances_per_image = {
        "mean": float(np.mean(n_inst_arr)),
        "std": float(np.std(n_inst_arr)),
        "min": int(np.min(n_inst_arr)),
        "max": int(np.max(n_inst_arr)),
    }

    # 面积统计
    all_areas = []
    all_bbox_ws = []
    all_bbox_hs = []
    all_aspects = []
    all_extents = []
    for s in stats_list:
        for inst in s.instances:
            all_areas.append(inst.area)
            all_bbox_ws.append(inst.bbox_w)
            all_bbox_hs.append(inst.bbox_h)
            all_aspects.append(inst.aspect_ratio)
            all_extents.append(inst.extent)

    area_arr = np.array(all_areas) if all_areas else np.zeros(0)
    area_stats = {
        "mean": float(np.mean(area_arr)) if len(area_arr) else 0.0,
        "std": float(np.std(area_arr)) if len(area_arr) else 0.0,
        "min": int(np.min(area_arr)) if len(area_arr) else 0,
        "max": int(np.max(area_arr)) if len(area_arr) else 0,
        "median": float(np.median(area_arr)) if len(area_arr) else 0.0,
        "bbox_w_mean": float(np.mean(all_bbox_ws)) if all_bbox_ws else 0.0,
        "bbox_h_mean": float(np.mean(all_bbox_hs)) if all_bbox_hs else 0.0,
        "aspect_ratio_mean": float(np.mean(all_aspects)) if all_aspects else 0.0,
        "extent_mean": float(np.mean(all_extents)) if all_extents else 0.0,
    }

    # 前景覆盖率
    fg_arr = np.array([s.fg_ratio for s in stats_list])
    fg_ratio_stats = {
        "mean": float(np.mean(fg_arr)),
        "std": float(np.std(fg_arr)),
        "min": float(np.min(fg_arr)),
        "max": float(np.max(fg_arr)),
    }

    # 面积分布 (COCO 尺度档次: tiny < 32², small < 96², medium < 256², large)
    area_dist = {
        "tiny (<32²)": int(np.sum(area_arr < 32 ** 2)),
        "small (32²-96²)": int(np.sum((area_arr >= 32 ** 2) & (area_arr < 96 ** 2))),
        "medium (96²-256²)": int(np.sum((area_arr >= 96 ** 2) & (area_arr < 256 ** 2))),
        "large (≥256²)": int(np.sum(area_arr >= 256 ** 2)),
    }

    # 逐图
    per_image = []
    for s in stats_list:
        per_image.append({
            "name": s.name,
            "source": s.source,
            "split": s.split,
            "n_instances": s.n_instances,
            "fg_pixels": s.fg_pixels,
            "fg_ratio": round(s.fg_ratio, 6),
            "instance_areas": [inst.area for inst in s.instances],
            "instance_aspect_ratios": [round(inst.aspect_ratio, 2) for inst in s.instances],
        })

    return DatasetReport(
        total_images=len(stats_list),
        total_instances=total_instances,
        image_size=(IMAGE_H, IMAGE_W),
        splits=dict(splits),
        sources=dict(sources),
        instances_per_image=instances_per_image,
        area_stats=area_stats,
        fg_ratio_stats=fg_ratio_stats,
        area_distribution=area_dist,
        per_image=per_image,
    )


# ═══════════════════════════════════════════════════════════════════
# 可视化 | Visualization
# ═══════════════════════════════════════════════════════════════════

def make_plots(report: DatasetReport, stats_list: list[ImageStats], out_dir: Path) -> None:
    """生成分析图表 (需要 matplotlib) | Generate analysis plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available, skipping plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 120})

    # ── 图 1: 实例数分布 + 前景覆盖率 ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1a. 每图实例数直方图
    n_inst_list = [s.n_instances for s in stats_list]
    axes[0].hist(n_inst_list, bins=range(0, max(n_inst_list) + 2), edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Instances per image")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Instances/Image (mean={report.instances_per_image['mean']:.1f})")

    # 1b. 实例面积分布 (log scale)
    all_areas = [inst.area for s in stats_list for inst in s.instances]
    if all_areas:
        axes[1].hist(all_areas, bins=30, edgecolor="black", alpha=0.7, color="orange")
        axes[1].set_xlabel("Instance area (pixels)")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Area Distribution (n={len(all_areas)})")
        axes[1].axvline(np.mean(all_areas), color="red", linestyle="--", label=f"mean={np.mean(all_areas):.0f}")
        axes[1].legend(fontsize=8)

    # 1c. 前景覆盖率分布
    fg_ratios = [s.fg_ratio * 100 for s in stats_list]
    axes[2].bar(range(len(fg_ratios)), sorted(fg_ratios), alpha=0.7, color="green")
    axes[2].set_xlabel("Image (sorted)")
    axes[2].set_ylabel("FG ratio (%)")
    axes[2].set_title(f"FG Coverage (mean={report.fg_ratio_stats['mean']*100:.2f}%)")
    axes[2].axhline(y=report.fg_ratio_stats['mean'] * 100, color="red", linestyle="--")

    fig.tight_layout()
    fig.savefig(out_dir / "distributions.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_dir / 'distributions.png'}")

    # ── 图 2: 面积散点图 (按 split 着色) + 长宽比 ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 2a. 面积 vs 长宽比 散点
    colors = {"train": "steelblue", "val": "darkorange"}
    for s in stats_list:
        areas = [inst.area for inst in s.instances]
        aspects = [inst.aspect_ratio for inst in s.instances]
        c = colors.get(s.split, "gray")
        axes[0].scatter(areas, aspects, c=c, alpha=0.6, s=30, label=s.split if s.split not in axes[0].get_legend_handles_labels()[1] else "")
    axes[0].set_xlabel("Area (pixels)")
    axes[0].set_ylabel("Aspect Ratio (max/min)")
    axes[0].set_title("Instance Area vs Aspect Ratio")
    axes[0].legend(fontsize=8)
    axes[0].set_xscale("log")

    # 2b. Train vs Val 对比
    for split_name, split_label in [("train", "Train"), ("val", "Val")]:
        subset = [s for s in stats_list if s.split == split_name]
        n_inst = [s.n_instances for s in subset]
        axes[1].bar(
            [f"{split_label}\n(n={len(subset)})"],
            [np.mean(n_inst) if n_inst else 0],
            yerr=[np.std(n_inst) if n_inst else 0],
            alpha=0.7, color=colors[split_name], capsize=8,
        )
    axes[1].set_ylabel("Mean instances/image")
    axes[1].set_title("Train vs Val Instance Count")

    fig.tight_layout()
    fig.savefig(out_dir / "area_aspect.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_dir / 'area_aspect.png'}")

    # ── 图 3: 逐图 FG 像素占比 (按 split 分色) ──
    fig, ax = plt.subplots(figsize=(14, 4))
    sorted_data = sorted(stats_list, key=lambda s: s.fg_ratio)
    names = [s.name for s in sorted_data]
    ratios = [s.fg_ratio * 100 for s in sorted_data]
    bar_colors = [colors[s.split] for s in sorted_data]
    ax.bar(range(len(names)), ratios, color=bar_colors, alpha=0.85)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("FG Ratio (%)")
    ax.set_title("Per-Image Foreground Coverage")
    # legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=colors["train"], label="Train"),
        Patch(color=colors["val"], label="Val"),
    ], fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "per_image_fg.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_dir / 'per_image_fg.png'}")


# ═══════════════════════════════════════════════════════════════════
# 终端输出 | Console output
# ═══════════════════════════════════════════════════════════════════

HEADER = "\033[1;36m"
SECTION = "\033[1;33m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def print_report(report: DatasetReport) -> None:
    """打印格式化报告 | Print formatted report."""
    s = report  # alias

    print(f"\n{HEADER}{'='*60}{RESET}")
    print(f"{HEADER}  NEU-SEG Dataset Analysis{RESET}")
    print(f"{HEADER}{'='*60}{RESET}")

    # ── Section 1: Basic stats ──
    print(f"\n{SECTION}[1] Basic Statistics{RESET}")
    print(f"  Images: {BOLD}{s.total_images}{RESET} ({s.image_size[0]} x {s.image_size[1]})")
    print(f"  Instances (CCs): {BOLD}{s.total_instances}{RESET}")
    print(f"  Splits: {s.splits}")
    print(f"  Sources: {s.splits['train']} train + {s.splits['val']} val")
    print(f"  Source types: SDI={s.sources.get('SDI', 0)}, SPDI={s.sources.get('SPDI', 0)}")

    # ── Section 2: Instance analysis ──
    print(f"\n{SECTION}[2] Instance Analysis{RESET}")
    ipi = s.instances_per_image
    print(f"  Instances/image: mean={ipi['mean']:.1f}, std={ipi['std']:.1f}, "
          f"min={ipi['min']}, max={ipi['max']}")
    print(f"  Images with 0 instances: {sum(1 for p in s.per_image if p['n_instances'] == 0)}")
    print(f"  Images with 1 instance:  {sum(1 for p in s.per_image if p['n_instances'] == 1)}")
    print(f"  Images with 2 instances: {sum(1 for p in s.per_image if p['n_instances'] == 2)}")
    print(f"  Images with 3+ instances:{sum(1 for p in s.per_image if p['n_instances'] >= 3)}")

    # ── Section 3: Area / scale ──
    print(f"\n{SECTION}[3] Area & Scale{RESET}")
    area = s.area_stats
    print(f"  Instance area (px): mean={area['mean']:.0f}, std={area['std']:.0f}, "
          f"min={area['min']}, max={area['max']}, median={area['median']:.0f}")
    print(f"  Bbox: mean w={area['bbox_w_mean']:.1f}, mean h={area['bbox_h_mean']:.1f}")
    print(f"  Aspect ratio (max/min): mean={area['aspect_ratio_mean']:.2f}")
    print(f"  Extent (area/bbox): mean={area['extent_mean']:.3f}")
    print(f"  {SECTION}Scale buckets (COCO-ish):{RESET}")
    for bucket, count in s.area_distribution.items():
        bar = "█" * max(1, count // max(1, s.total_instances // 20))
        print(f"    {bucket:<20} {count:>3}  {DIM}{bar}{RESET}")

    # ── Section 4: Foreground imbalance ──
    print(f"\n{SECTION}[4] Foreground Coverage (Class Imbalance){RESET}")
    fg = s.fg_ratio_stats
    print(f"  FG ratio: mean={fg['mean']*100:.3f}%, std={fg['std']*100:.3f}%, "
          f"min={fg['min']*100:.3f}%, max={fg['max']*100:.3f}%")
    bg_ratio = (1 - fg['mean']) * 100
    print(f"  BG:FG ratio = {bg_ratio:.1f}:{fg['mean']*100:.2f}  "
          f"({DIM}extreme imbalance, focal gamma=5.0 is appropriate{RESET})")
    print(f"  Images with FG < 0.1%: {sum(1 for p in s.per_image if p['fg_ratio'] < 0.001)}")
    print(f"  Images with FG < 0.5%: {sum(1 for p in s.per_image if p['fg_ratio'] < 0.005)}")

    # ── Section 5: Train/Val comparison ──
    print(f"\n{SECTION}[5] Train/Val Split Comparison{RESET}")
    for split_name in ["train", "val"]:
        subset = [p for p in s.per_image if p["split"] == split_name]
        n_imgs = len(subset)
        n_inst = sum(p["n_instances"] for p in subset)
        fg_mean = np.mean([p["fg_ratio"] for p in subset]) if subset else 0
        print(f"  {split_name:>5}: {n_imgs:>2} images, {n_inst:>2} instances, "
              f"mean FG={fg_mean*100:.3f}%")

    # ── Section 6: Few-shot feasibility ──
    print(f"\n{SECTION}[6] Few-Shot Feasibility{RESET}")
    print(f"  Single class (building) — episode sampling always valid")
    print(f"  Scene-disjoint constraint: each image is its own scene")
    print(f"  Min K-shot recommendation: K={min(3, s.splits['train'] // 2)}")
    print(f"  Max episodes/epoch: ~{s.splits['train'] * (s.splits['train'] - 1)} possible pairs")

    print(f"\n{HEADER}{'='*60}{RESET}\n")


# ═══════════════════════════════════════════════════════════════════
# 入口 | Entry point
# ═══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NEU-SEG Dataset Analysis")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                   help="Neu_seg dataset root directory")
    p.add_argument("--min-area", type=int, default=4,
                   help="minimum CC area to count as instance")
    p.add_argument("--output-dir", default=None,
                   help="output directory for report and plots")
    p.add_argument("--plots", action="store_true",
                   help="generate matplotlib plots")
    p.add_argument("--json-only", action="store_true",
                   help="only write JSON, skip console output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.data_root)
    out_dir = Path(args.output_dir) if args.output_dir else (
        _REPO_ROOT / "runs" / "neuseg_analysis"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 读取 split ──
    train_names = load_split_set(root, "train")
    val_names = load_split_set(root, "val")

    # ── 分析每张图 ──
    img_dir = root / "img_dir"
    ann_dir = root / "ann_dir"
    stats_list: list[ImageStats] = []

    for img_path in sorted(img_dir.glob("*.jpg")):
        stem = img_path.stem
        ann_path = ann_dir / f"{stem}.png"
        if not ann_path.exists():
            print(f"[skip] no annotation for {stem}")
            continue

        if stem in train_names:
            split = "train"
        elif stem in val_names:
            split = "val"
        else:
            split = "unknown"

        source = "SDI" if stem.startswith("SDI") else "SPDI"
        img_stats = analyze_image(img_path, ann_path, stem, split, source, args.min_area)
        if img_stats:
            stats_list.append(img_stats)

    # ── 构建报告 ──
    report = build_report(stats_list)

    # ── 输出 ──
    if not args.json_only:
        print_report(report)

    # JSON 报告
    report_path = out_dir / "neu_seg_report.json"
    report_dict = {
        "dataset": "NEU-SEG",
        "total_images": report.total_images,
        "total_instances": report.total_instances,
        "image_size": list(report.image_size),
        "splits": report.splits,
        "sources": report.sources,
        "instances_per_image": report.instances_per_image,
        "area_stats": report.area_stats,
        "fg_ratio_stats": report.fg_ratio_stats,
        "area_distribution": report.area_distribution,
        "per_image": report.per_image,
    }
    report_path.write_text(json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[report] {report_path}")

    # 图表
    if args.plots:
        make_plots(report, stats_list, out_dir)

    print(f"[done] Output directory: {out_dir}")


if __name__ == "__main__":
    main()
