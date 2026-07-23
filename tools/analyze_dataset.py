"""
数据集统计分析 | Dataset statistics analysis.
=============================================

对切分后的 iSAID Instance Few-Shot 数据集 (896² COCO tiles) 做逐类分析:
① 类别分布 (实例/tile/场景数, min_tiles 过滤, K-shot 可行性)
② 尺度 (COCO S/M/L 三档, 64² 特征网格占格数, 极小实例)
③ 密度 (每 tile 单类实例数分布, 超 num_queries 的 tile 占比)
④ 结构质量 (贴边截断, iscrowd, 类共现, 场景 tile 数, 长宽比/填充率)
⑤ fold 划分 (3-fold base/novel 实例量均衡性)

Per-class analysis of the tiled iSAID Instance Few-Shot dataset (896² COCO tiles):
class distribution / scale / density / structure quality / fold splits.
仅读 COCO JSON, 不加载图像与掩码 (面积取标注 area 字段, 与加载器的渲染面积略有出入)。
Reads COCO JSON only; never loads images or renders masks (areas come from the
annotation "area" field and may differ slightly from the loader's rendered areas).

输出 | Output:
    - 终端汇总表 | console summary tables
    - <output>/report.json  (全量数字 | full numbers)
    - <output>/*.png        (--plots 时 | with --plots)

用法 | Usage::

    python tools/analyze_dataset.py                    # data_root from configs/base.yaml
    python tools/analyze_dataset.py --data-root /root/autodl-tmp/iSAID_instance_fewshot --plots
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Windows GBK 控制台无法打印 ²/→ 等字符 | Windows GBK console can't print ²/→ etc.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from adasam.datasets.isaid import DEFAULT_FOLDS, ISAID_CATEGORIES, MIN_INSTANCE_AREA  # noqa: E402

# COCO 尺度阈值 | COCO size thresholds (area in px²)
COCO_SMALL = 32 * 32  # < 1024 → small
COCO_MEDIUM = 96 * 96  # < 9216 → medium, else large
GRID = 64  # DPG / 相似度特征网格边长 | feature-grid side length


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AdaSAM dataset statistics analyzer")
    p.add_argument("--config", default=str(_REPO_ROOT / "configs" / "base.yaml"))
    p.add_argument("--data-root", default=None, help="override data.data_root")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="density threshold N (default: prompt_generator.num_queries)",
    )
    p.add_argument("--k-shot", type=int, default=None, help="default: fewshot.k_shot")
    p.add_argument("--min-tiles", type=int, default=None, help="default: fewshot.min_tiles")
    p.add_argument("--plots", action="store_true", help="save matplotlib PNGs")
    p.add_argument("--output-dir", default=None, help="default: runs/dataset_analysis")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# 数据装载 | Split loading
# ═══════════════════════════════════════════════════════════════════


@dataclass
class SplitData:
    """单个 split 的轻量索引 | lightweight per-split indices (no images)."""

    split: str
    tiles: dict[int, dict] = field(default_factory=dict)  # image_id → record
    anns: list[dict] = field(default_factory=list)
    anns_by_tile: dict[int, list[dict]] = field(default_factory=dict)
    class_to_tiles: dict[int, set[int]] = field(default_factory=dict)
    tile_scene: dict[int, int] = field(default_factory=dict)  # image_id → orig_image_id


def load_split(data_root: Path, split: str) -> SplitData:
    """从 COCO JSON 建索引 | build indices from the COCO JSON."""
    anno_path = data_root / "annotations" / f"instances_{split}.json"
    if not anno_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {anno_path}")
    with open(anno_path, encoding="utf-8") as f:
        coco = json.load(f)

    d = SplitData(split=split)
    d.tiles = {int(img["id"]): img for img in coco.get("images", [])}
    d.tile_scene = {tid: int(img.get("orig_image_id", tid)) for tid, img in d.tiles.items()}
    d.anns = [a for a in coco.get("annotations", []) if 1 <= a.get("category_id", 0) <= 15]

    anns_by_tile: dict[int, list[dict]] = defaultdict(list)
    class_to_tiles: dict[int, set[int]] = defaultdict(set)
    for a in d.anns:
        anns_by_tile[a["image_id"]].append(a)
        class_to_tiles[a["category_id"]].add(a["image_id"])
    d.anns_by_tile = dict(anns_by_tile)
    d.class_to_tiles = dict(class_to_tiles)
    return d


def _area(ann: dict) -> float:
    """标注面积, 缺失时回退 bbox 面积 | annotation area, falling back to bbox area."""
    a = ann.get("area")
    if a is not None and a > 0:
        return float(a)
    _, _, w, h = ann.get("bbox", [0, 0, 0, 0])
    return float(w * h)


# ═══════════════════════════════════════════════════════════════════
# ① 类别分布 | Class distribution
# ═══════════════════════════════════════════════════════════════════


def class_distribution(d: SplitData, min_tiles: int, k_shot: int) -> dict:
    out: dict[str, dict] = {}
    for cls in sorted(ISAID_CATEGORIES):
        tiles = d.class_to_tiles.get(cls, set())
        n_inst = sum(1 for a in d.anns if a["category_id"] == cls)
        scenes = {d.tile_scene[t] for t in tiles}
        out[str(cls)] = {
            "name": ISAID_CATEGORIES[cls],
            "n_instances": n_inst,
            "n_tiles": len(tiles),
            "n_scenes": len(scenes),
            "excluded_by_min_tiles": len(tiles) < min_tiles,
            # K 个 support + ≥1 个 query 需 K+1 个互异场景 | K supports + ≥1 query need K+1 scenes
            "kshot_scene_disjoint_ok": len(scenes) >= k_shot + 1,
        }
    return out


# ═══════════════════════════════════════════════════════════════════
# ② 尺度 | Scale
# ═══════════════════════════════════════════════════════════════════


def scale_analysis(d: SplitData, tile_size: int) -> dict:
    cell_px = tile_size / GRID  # 一个特征格对应的 tile 像素 | px per grid cell
    cell_area = cell_px * cell_px  # 896/64=14 → 196 px²
    out: dict[str, dict] = {"_cell_area_px": cell_area}
    areas_by_cls: dict[int, list[float]] = defaultdict(list)
    for a in d.anns:
        areas_by_cls[a["category_id"]].append(_area(a))
    for cls in sorted(ISAID_CATEGORIES):
        arr = np.asarray(areas_by_cls.get(cls, []), dtype=np.float64)
        if arr.size == 0:
            continue
        out[str(cls)] = {
            "name": ISAID_CATEGORIES[cls],
            "area_min": float(arr.min()),
            "area_median": float(np.median(arr)),
            "area_max": float(arr.max()),
            "pct_small": float((arr < COCO_SMALL).mean() * 100),
            "pct_medium": float(((arr >= COCO_SMALL) & (arr < COCO_MEDIUM)).mean() * 100),
            "pct_large": float((arr >= COCO_MEDIUM).mean() * 100),
            "pct_below_1cell": float((arr < cell_area).mean() * 100),
            "median_grid_cells": float(np.median(arr) / cell_area),
            "n_below_min_area": int((arr < MIN_INSTANCE_AREA).sum()),  # 加载器丢弃 | loader drops
        }
    return out


# ═══════════════════════════════════════════════════════════════════
# ③ 密度 | Density
# ═══════════════════════════════════════════════════════════════════


def density_analysis(d: SplitData, num_queries: int) -> dict:
    per_tile_cls: dict[int, list[int]] = defaultdict(list)  # cls → [该类每 tile 实例数]
    per_tile_total: list[int] = []
    for anns in d.anns_by_tile.values():
        per_tile_total.append(len(anns))
        counts: dict[int, int] = defaultdict(int)
        for a in anns:
            counts[a["category_id"]] += 1
        for cls, n in counts.items():
            per_tile_cls[cls].append(n)

    out: dict[str, dict] = {}
    for cls in sorted(ISAID_CATEGORIES):
        arr = np.asarray(per_tile_cls.get(cls, []), dtype=np.int64)
        if arr.size == 0:
            continue
        n_over = int((arr > num_queries).sum())
        out[str(cls)] = {
            "name": ISAID_CATEGORIES[cls],
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "max": int(arr.max()),
            "n_tiles_over_N": n_over,  # 训练截断/推理召回天花板 | recall cap
            "pct_tiles_over_N": float(n_over / arr.size * 100),
        }
    total = np.asarray(per_tile_total, dtype=np.int64)
    out["_per_tile_total"] = {
        "n_tiles_with_anns": int(total.size),
        "mean": float(total.mean()) if total.size else 0.0,
        "median": float(np.median(total)) if total.size else 0.0,
        "p95": float(np.percentile(total, 95)) if total.size else 0.0,
        "max": int(total.max()) if total.size else 0,
    }
    return out


# ═══════════════════════════════════════════════════════════════════
# ④ 结构质量 | Structure & quality
# ═══════════════════════════════════════════════════════════════════


def structure_analysis(d: SplitData) -> dict:
    eps = 1.0
    border_by_cls: dict[int, list[bool]] = defaultdict(list)
    elong_by_cls: dict[int, list[float]] = defaultdict(list)
    fill_by_cls: dict[int, list[float]] = defaultdict(list)
    n_crowd = 0
    for a in d.anns:
        if a.get("iscrowd", 0):
            n_crowd += 1
        img = d.tiles.get(a["image_id"], {})
        tw, th = float(img.get("width", 896)), float(img.get("height", 896))
        x, y, w, h = (float(v) for v in a.get("bbox", [0, 0, 0, 0]))
        cls = a["category_id"]
        border_by_cls[cls].append(x <= eps or y <= eps or x + w >= tw - eps or y + h >= th - eps)
        if w > 0 and h > 0:
            elong_by_cls[cls].append(max(w, h) / min(w, h))
            fill_by_cls[cls].append(_area(a) / (w * h))

    per_class: dict[str, dict] = {}
    for cls in sorted(ISAID_CATEGORIES):
        if cls not in border_by_cls:
            continue
        per_class[str(cls)] = {
            "name": ISAID_CATEGORIES[cls],
            "pct_border": float(np.mean(border_by_cls[cls]) * 100),
            "median_elongation": float(np.median(elong_by_cls[cls])) if elong_by_cls[cls] else 0.0,
            "median_fill": float(np.median(fill_by_cls[cls])) if fill_by_cls[cls] else 0.0,
        }

    # 类共现: 同 tile 出现的 (类, 类) tile 数 | class co-occurrence tile counts
    cooc = np.zeros((15, 15), dtype=np.int64)
    for anns in d.anns_by_tile.values():
        present = sorted({a["category_id"] for a in anns})
        for i, ci in enumerate(present):
            for cj in present[i:]:
                cooc[ci - 1, cj - 1] += 1
                if ci != cj:
                    cooc[cj - 1, ci - 1] += 1

    scene_tiles: dict[int, int] = defaultdict(int)
    for tid in d.tiles:
        scene_tiles[d.tile_scene[tid]] += 1
    st = np.asarray(list(scene_tiles.values()), dtype=np.int64)

    all_border = [b for lst in border_by_cls.values() for b in lst]
    return {
        "per_class": per_class,
        "pct_border_overall": float(np.mean(all_border) * 100) if all_border else 0.0,
        "n_iscrowd": n_crowd,
        "cooccurrence": cooc.tolist(),
        "scenes": {
            "n_scenes": int(st.size),
            "tiles_per_scene_mean": float(st.mean()) if st.size else 0.0,
            "tiles_per_scene_median": float(np.median(st)) if st.size else 0.0,
            "tiles_per_scene_max": int(st.max()) if st.size else 0,
        },
    }


# ═══════════════════════════════════════════════════════════════════
# ⑤ Fold 划分 | Fold splits
# ═══════════════════════════════════════════════════════════════════


def fold_analysis(data_root: Path, dist_by_split: dict[str, dict]) -> dict:
    out: dict[str, dict] = {}
    for fold in (0, 1, 2):
        fold_path = data_root / "folds" / f"fold_{fold}.json"
        if fold_path.exists():
            with open(fold_path, encoding="utf-8") as f:
                fd = json.load(f)
        else:
            fd = DEFAULT_FOLDS[fold]
        entry: dict = {"base": list(fd["base"]), "novel": list(fd["novel"])}
        for split, dist in dist_by_split.items():
            entry[f"{split}_base_instances"] = sum(
                dist[str(c)]["n_instances"] for c in fd["base"] if str(c) in dist
            )
            entry[f"{split}_novel_instances"] = sum(
                dist[str(c)]["n_instances"] for c in fd["novel"] if str(c) in dist
            )
        out[str(fold)] = entry
    return out


# ═══════════════════════════════════════════════════════════════════
# 终端打印 | Console printing
# ═══════════════════════════════════════════════════════════════════


def _print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)
    ]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(f"\n── {title} ──")
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(v.ljust(w) for v, w in zip(r, widths)))


def _sorted_cls(section: dict, key: str) -> list[str]:
    """按指标降序的类 id 列表 (跳过 _ 前缀) | class ids sorted by metric desc (skip _keys)."""
    ids = [c for c in section if not c.startswith("_")]
    return sorted(ids, key=lambda c: section[c].get(key, 0), reverse=True)


def print_split_report(split: str, rep: dict, num_queries: int) -> None:
    print(f"\n{'=' * 78}\n  SPLIT: {split}\n{'=' * 78}")

    dist = rep["class_distribution"]
    rows = []
    for c in _sorted_cls(dist, "n_instances"):
        r = dist[c]
        flags = []
        if r["excluded_by_min_tiles"]:
            flags.append("min_tiles!")
        if not r["kshot_scene_disjoint_ok"]:
            flags.append("k-shot!")
        rows.append(
            [
                c,
                r["name"],
                f"{r['n_instances']:,}",
                f"{r['n_tiles']:,}",
                str(r["n_scenes"]),
                " ".join(flags),
            ]
        )
    _print_table(
        "① 类别分布 | class distribution",
        ["id", "class", "instances", "tiles", "scenes", "flags"],
        rows,
    )

    sc = rep["scale"]
    rows = []
    for c in _sorted_cls(sc, "pct_small"):
        r = sc[c]
        rows.append(
            [
                c,
                r["name"],
                f"{r['area_median']:.0f}",
                f"{r['pct_small']:.1f}",
                f"{r['pct_medium']:.1f}",
                f"{r['pct_large']:.1f}",
                f"{r['pct_below_1cell']:.1f}",
                f"{r['median_grid_cells']:.1f}",
            ]
        )
    _print_table(
        f"② 尺度 | scale (1 grid cell = {sc['_cell_area_px']:.0f} px²)",
        ["id", "class", "med_area", "S%", "M%", "L%", "<1cell%", "med_cells"],
        rows,
    )

    de = rep["density"]
    rows = []
    for c in _sorted_cls(de, "p95"):
        r = de[c]
        rows.append(
            [
                c,
                r["name"],
                f"{r['mean']:.1f}",
                f"{r['median']:.0f}",
                f"{r['p95']:.0f}",
                str(r["max"]),
                f"{r['pct_tiles_over_N']:.1f} ({r['n_tiles_over_N']})",
            ]
        )
    _print_table(
        f"③ 密度 | per-tile per-class instance counts (N={num_queries})",
        ["id", "class", "mean", "med", "p95", "max", f">{num_queries}% (tiles)"],
        rows,
    )
    t = de["_per_tile_total"]
    print(
        f"  per-tile TOTAL: mean={t['mean']:.1f} med={t['median']:.0f} "
        f"p95={t['p95']:.0f} max={t['max']}  (tiles with anns: {t['n_tiles_with_anns']:,})"
    )

    st = rep["structure"]
    rows = []
    for c in _sorted_cls(st["per_class"], "pct_border"):
        r = st["per_class"][c]
        rows.append(
            [
                c,
                r["name"],
                f"{r['pct_border']:.1f}",
                f"{r['median_elongation']:.2f}",
                f"{r['median_fill']:.2f}",
            ]
        )
    _print_table("④ 结构 | structure", ["id", "class", "border%", "elong", "fill"], rows)
    print(
        f"  overall border-touching: {st['pct_border_overall']:.1f}%  "
        f"iscrowd: {st['n_iscrowd']}  scenes: {st['scenes']['n_scenes']} "
        f"(tiles/scene med={st['scenes']['tiles_per_scene_median']:.0f} "
        f"max={st['scenes']['tiles_per_scene_max']})"
    )

    cooc = np.asarray(st["cooccurrence"])
    pairs = [
        (int(cooc[i, j]), i + 1, j + 1)
        for i in range(15)
        for j in range(i + 1, 15)
        if cooc[i, j] > 0
    ]
    pairs.sort(reverse=True)
    print("  top co-occurring pairs:")
    for n, ci, cj in pairs[:8]:
        print(f"    {ISAID_CATEGORIES[ci]} + {ISAID_CATEGORIES[cj]}: {n:,} tiles")


def print_fold_report(folds: dict, splits: list[str]) -> None:
    print(f"\n{'=' * 78}\n  ⑤ FOLDS (base/novel)\n{'=' * 78}")
    for fold, e in folds.items():
        novel_names = ", ".join(ISAID_CATEGORIES[c] for c in e["novel"])
        print(f"\n  fold {fold}: novel = [{novel_names}]")
        for split in splits:
            b, n = e.get(f"{split}_base_instances", 0), e.get(f"{split}_novel_instances", 0)
            print(f"    {split}: base={b:,}  novel={n:,}")


# ═══════════════════════════════════════════════════════════════════
# 可选出图 | Optional plots
# ═══════════════════════════════════════════════════════════════════


def make_plots(report: dict, out_dir: Path, splits: list[str], num_queries: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze] matplotlib missing; skip plots (pip install matplotlib)")
        return

    names = [ISAID_CATEGORIES[c] for c in sorted(ISAID_CATEGORIES)]
    idx = np.arange(15)

    # 类分布 (各 split 同图) | class distribution, all splits in one figure
    fig, ax = plt.subplots(figsize=(12, 4.5))
    width = 0.8 / max(1, len(splits))
    for si, split in enumerate(splits):
        dist = report[split]["class_distribution"]
        vals = [dist.get(str(c), {}).get("n_instances", 0) for c in sorted(ISAID_CATEGORIES)]
        ax.bar(idx + si * width, vals, width, label=split)
    ax.set_yscale("log")
    ax.set_xticks(idx + width * (len(splits) - 1) / 2)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("instances (log)")
    ax.set_title("Instances per class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    for split in splits:
        rep = report[split]

        # 尺度堆叠条 | stacked S/M/L bars
        sc = rep["scale"]
        s = np.array([sc.get(str(c), {}).get("pct_small", 0) for c in sorted(ISAID_CATEGORIES)])
        m = np.array([sc.get(str(c), {}).get("pct_medium", 0) for c in sorted(ISAID_CATEGORIES)])
        lg = np.array([sc.get(str(c), {}).get("pct_large", 0) for c in sorted(ISAID_CATEGORIES)])
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.bar(idx, s, label="small <32²", color="#d62728")
        ax.bar(idx, m, bottom=s, label="medium", color="#ff7f0e")
        ax.bar(idx, lg, bottom=s + m, label="large ≥96²", color="#2ca02c")
        ax.set_xticks(idx)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("%")
        ax.set_title(f"COCO size buckets ({split})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"size_buckets_{split}.png", dpi=150)
        plt.close(fig)

        # 密度 p95/max 与 N 线 | density p95/max vs the num_queries line
        de = rep["density"]
        p95 = [de.get(str(c), {}).get("p95", 0) for c in sorted(ISAID_CATEGORIES)]
        mx = [de.get(str(c), {}).get("max", 0) for c in sorted(ISAID_CATEGORIES)]
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.bar(idx - 0.2, p95, 0.4, label="P95")
        ax.bar(idx + 0.2, mx, 0.4, label="max")
        ax.axhline(num_queries, color="red", ls="--", label=f"N={num_queries}")
        ax.set_yscale("log")
        ax.set_xticks(idx)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("instances/tile (log)")
        ax.set_title(f"Per-tile per-class density ({split})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"density_{split}.png", dpi=150)
        plt.close(fig)

        # 共现热图 | co-occurrence heatmap
        cooc = np.asarray(rep["structure"]["cooccurrence"], dtype=np.float64)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(np.log1p(cooc), cmap="viridis")
        ax.set_xticks(idx)
        ax.set_xticklabels(names, rotation=90)
        ax.set_yticks(idx)
        ax.set_yticklabels(names)
        ax.set_title(f"Class co-occurrence, log(1+tiles) ({split})")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(out_dir / f"cooccurrence_{split}.png", dpi=150)
        plt.close(fig)

    print(f"[analyze] plots saved → {out_dir}")


# ═══════════════════════════════════════════════════════════════════
# 主流程 | Main
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_root = Path(args.data_root or cfg["data"]["data_root"])
    tile_size = int(cfg["data"].get("tile_size", 896))
    num_queries = args.num_queries or int(cfg.get("prompt_generator", {}).get("num_queries", 64))
    k_shot = args.k_shot or int(cfg.get("fewshot", {}).get("k_shot", 5))
    min_tiles = args.min_tiles or int(cfg.get("fewshot", {}).get("min_tiles", 30))
    out_dir = (
        Path(args.output_dir) if args.output_dir else (_REPO_ROOT / "runs" / "dataset_analysis")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analyze] data_root = {data_root}")
    print(
        f"[analyze] N={num_queries}  k_shot={k_shot}  min_tiles={min_tiles}  "
        f"grid={GRID}²  tile={tile_size}²"
    )

    report: dict = {
        "meta": {
            "data_root": str(data_root),
            "tile_size": tile_size,
            "grid": GRID,
            "num_queries": num_queries,
            "k_shot": k_shot,
            "min_tiles": min_tiles,
            "min_instance_area": MIN_INSTANCE_AREA,
        },
    }
    dist_by_split: dict[str, dict] = {}
    for split in args.splits:
        d = load_split(data_root, split)
        rep = {
            "n_tiles": len(d.tiles),
            "n_instances": len(d.anns),
            "class_distribution": class_distribution(d, min_tiles, k_shot),
            "scale": scale_analysis(d, tile_size),
            "density": density_analysis(d, num_queries),
            "structure": structure_analysis(d),
        }
        report[split] = rep
        dist_by_split[split] = rep["class_distribution"]
        print(f"\n[analyze] {split}: {rep['n_tiles']:,} tiles, {rep['n_instances']:,} instances")
        print_split_report(split, rep, num_queries)

    report["folds"] = fold_analysis(data_root, dist_by_split)
    print_fold_report(report["folds"], args.splits)

    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[analyze] report → {report_path}")

    if args.plots:
        make_plots(report, out_dir, args.splits, num_queries)


if __name__ == "__main__":
    main()
