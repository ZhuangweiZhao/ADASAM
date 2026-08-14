"""Export a consolidated summary CSV of every run under runs/云服务器.

Sources:
  1. Every ``metrics.json`` under results/ and legacy/ (excluding .ipynb_checkpoints)
  2. Top-level single-run JSON files (LoveDA_*.json, iSAID_full.json)
  3. logs/multiseed_summary.json aggregates (marked record_type=aggregate)

Output: runs/云服务器/云服务器结果汇总.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

CLOUD = Path(r"runs/云服务器")
OUT = CLOUD / "云服务器结果汇总.csv"

# columns: (csv_name, source_path_or_None, default)
COLUMNS = [
    ("record_type", None, "run"),
    ("source", None, ""),
    ("suite", None, ""),
    ("dataset", "dataset", ""),
    ("model", "model", ""),
    ("variant", None, ""),
    ("ratio", None, ""),
    ("seed", None, ""),
    ("augmentation", "augmentation", ""),
    ("adapter", "adapter", ""),
    ("adapter_placement", "adapter_placement", ""),
    ("feature_scales", "feature_scales", ""),
    ("fusion_version", "fusion_version", ""),
    ("representation_budget", None, ""),
    ("decoder_version", "decoder_version", ""),
    ("boundary_loss_weight", "boundary_loss_weight", ""),
    ("label_pool_samples", "label_pool_samples", ""),
    ("train_samples", "train_samples", ""),
    ("validation_samples", "validation_samples", ""),
    ("test_samples", None, ""),
    ("params_total", "parameters.total", ""),
    ("params_trainable", "parameters.trainable", ""),
    ("params_frozen", "parameters.frozen", ""),
    ("best_epoch", "best_epoch", ""),
    ("epochs", None, ""),
    ("max_iterations", None, ""),
    ("train_time_seconds", None, ""),
    ("image_size", None, ""),
    ("sam_image_size", None, ""),
    ("batch_size", None, ""),
    ("lr", None, ""),
    ("mIoU", "test.mIoU", ""),
    ("mIoU_fg", "test.mIoU_fg", ""),
    ("Dice", "test.Dice", ""),
    ("Dice_fg", "test.Dice_fg", ""),
    ("pixel_accuracy", "test.pixel_accuracy", ""),
    ("Boundary_F1", "test.Boundary_F1", ""),
    ("Boundary_precision", "test.Boundary_precision", ""),
    ("Boundary_recall", "test.Boundary_recall", ""),
    ("FPS", "test.FPS", ""),
    ("test_seconds", "test.seconds", ""),
    ("per_class_IoU", "test.per_class_IoU", ""),
    ("per_class_Dice", "test.per_class_Dice", ""),
    ("routing_P3", "routing_statistics.mean_weights.P3", ""),
    ("routing_P4", "routing_statistics.mean_weights.P4", ""),
    ("routing_embedding", "routing_statistics.mean_weights.embedding", ""),
    ("routing_entropy", "routing_statistics.mean_entropy", ""),
    # aggregate-only columns
    ("n_runs", None, ""),
    ("mIoU_mean", None, ""),
    ("mIoU_std", None, ""),
    ("mIoU_fg_mean", None, ""),
    ("mIoU_fg_std", None, ""),
    ("Dice_mean", None, ""),
    ("Dice_std", None, ""),
    ("FPS_mean", None, ""),
    ("FPS_std", None, ""),
    ("train_time_mean", None, ""),
    ("train_time_std", None, ""),
]


def deep_get(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def fmt(v, precision: int = 6) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{precision}f}"
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    return str(v)


def guess_dataset(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    if any("isaid" in p for p in parts):
        return "iSAID"
    if any("loveda" in p or "love" in p for p in parts):
        return "LoveDA"
    if any("neu" in p for p in parts):
        return "NEU_Seg"
    return ""


def guess_seed(text: str) -> str:
    import re

    m = re.search(r"seed(\d+)", text)
    return m.group(1) if m else ""


def guess_ratio(text: str) -> str:
    import re

    m = re.search(r"ratio(\d+)", text)
    return m.group(1) if m else ""


def suite_from_path(path: Path) -> str:
    """Derive a short suite name from the run folder path."""
    import re

    rel = path.relative_to(CLOUD)
    parts = [p for p in rel.parts if p not in (".ipynb_checkpoints", "metrics.json")]
    # drop the trailing run folder (e.g. loveda_ratio10_seed42 / neu_seg_ratio10_seed42)
    if len(parts) > 1:
        parts = parts[:-1]
    # strip ratio/seed markers from the final folder for a clean suite name
    if parts:
        parts[-1] = re.sub(r"_ratio\d+_seed\d+$", "", parts[-1])
        parts[-1] = re.sub(r"_seed\d+$", "", parts[-1])
    return "/".join(parts)


def collect_metrics_files() -> list[Path]:
    found = []
    for base in (CLOUD / "results", CLOUD / "legacy"):
        if base.exists():
            for p in base.rglob("metrics.json"):
                if ".ipynb_checkpoints" in p.parts:
                    continue
                found.append(p)
    return found


def blank_row() -> dict:
    return {c[0]: "" for c in COLUMNS}


def run_row_from_metrics(path: Path) -> dict:
    d = json.load(open(path, encoding="utf-8"))
    row = blank_row()
    row["record_type"] = "run"
    row["source"] = str(path.relative_to(CLOUD))
    for csv_name, src, _ in COLUMNS:
        if src:
            v = deep_get(d, src)
            if v is not None:
                row[csv_name] = fmt(v)
    args = d.get("args") or {}
    # args overrides for config fields
    arg_map = {
        "epochs": "epochs",
        "max_iterations": "max_iterations",
        "image_size": "image_size",
        "sam_image_size": "sam_image_size",
        "batch_size": "batch_size",
        "lr": "lr",
        "representation_budget": "representation_budget",
        "augmentation": "augmentation",
        "adapter": "adapter",
        "feature_scales": "feature_scales",
        "fusion_version": "fusion_version",
        "decoder_version": "decoder_version",
    }
    for csv_name, arg_key in arg_map.items():
        if arg_key in args and not row.get(csv_name):
            row[csv_name] = fmt(args[arg_key])
    # NEU uses img_size instead of image_size
    if not row["image_size"] and "img_size" in args:
        row["image_size"] = fmt(args["img_size"])
    # dataset / model / ratio / seed fallbacks
    if not row["dataset"]:
        row["dataset"] = guess_dataset(path)
    if not row["model"]:
        for key in ("model",):
            if key in args:
                row["model"] = fmt(args[key])
    if not row["ratio"]:
        row["ratio"] = guess_ratio(str(path)) or fmt(d.get("label_ratio"))
    if not row["seed"]:
        row["seed"] = guess_seed(str(path)) or fmt(args.get("seed"))
    if not row["train_samples"]:
        row["train_samples"] = fmt(d.get("train_samples"))
    if not row["label_pool_samples"]:
        row["label_pool_samples"] = fmt(d.get("label_pool_samples"))
    if not row["test_samples"]:
        row["test_samples"] = fmt(deep_get(d, "test.samples"))
    # train time from history
    if not row["train_time_seconds"]:
        hist = d.get("history") or []
        total = sum(float(e.get("seconds", 0.0)) for e in hist if isinstance(e, dict))
        row["train_time_seconds"] = fmt(total, 2) if total else ""
    row["suite"] = suite_from_path(path)
    row["variant"] = row["suite"].rsplit("/", 1)[-1]
    return row


def add_single_run_file(path: Path, variant: str, dataset: str) -> dict:
    d = json.load(open(path, encoding="utf-8"))
    row = blank_row()
    row["record_type"] = "run"
    row["source"] = str(path.relative_to(CLOUD))
    row["variant"] = variant
    for csv_name, src, _ in COLUMNS:
        if src:
            v = deep_get(d, src)
            if v is not None:
                row[csv_name] = fmt(v)
    args = d.get("args") or {}
    if not row["model"]:
        row["model"] = fmt(d.get("model")) or fmt(args.get("model"))
    if not row["dataset"]:
        row["dataset"] = dataset
    if not row["ratio"]:
        row["ratio"] = guess_ratio(str(path)) or fmt(d.get("label_ratio"))
    if not row["seed"]:
        row["seed"] = fmt(d.get("seed")) or fmt(args.get("seed"))
    if not row["train_samples"]:
        row["train_samples"] = fmt(d.get("train_samples"))
    if not row["test_samples"]:
        row["test_samples"] = fmt(deep_get(d, "test.samples"))
    hist = d.get("history") or []
    total = sum(float(e.get("seconds", 0.0)) for e in hist if isinstance(e, dict))
    row["train_time_seconds"] = fmt(total, 2) if total else ""
    row["suite"] = variant
    return row


def aggregate_rows_from_multiseed() -> list[dict]:
    p = CLOUD / "logs/multiseed_summary.json"
    d = json.load(open(p, encoding="utf-8"))
    rows = []
    for a in d.get("aggregates", []):
        row = {
            "record_type": "aggregate",
            "source": "logs/multiseed_summary.json",
            "suite": f"NEU_Seg_{a.get('model')}_{a.get('augmentation')}",
            "dataset": d.get("protocol", {}).get("dataset", "NEU_Seg"),
            "model": a.get("model", ""),
            "augmentation": a.get("augmentation", ""),
            "ratio": a.get("ratio", ""),
            "n_runs": a.get("runs", ""),
            "mIoU_mean": fmt(a.get("mIoU_mean")),
            "mIoU_std": fmt(a.get("mIoU_std")),
            "mIoU_fg_mean": fmt(a.get("mIoU_fg_mean")),
            "mIoU_fg_std": fmt(a.get("mIoU_fg_std")),
            "Dice_mean": fmt(a.get("Dice_mean")),
            "Dice_std": fmt(a.get("Dice_std")),
            "FPS_mean": fmt(a.get("FPS_mean")),
            "FPS_std": fmt(a.get("FPS_std")),
            "train_time_mean": fmt(a.get("train_time_seconds_mean")),
            "train_time_std": fmt(a.get("train_time_seconds_std")),
        }
        rows.append(row)
    return rows


def main() -> None:
    rows: list[dict] = []

    # 1. per-run metrics.json
    for p in collect_metrics_files():
        try:
            rows.append(run_row_from_metrics(p))
        except Exception as e:  # noqa: BLE001
            print(f"WARN: failed to parse {p}: {e}")

    # 2. top-level single-run files
    singles = [
        ("results/loveda/LoveDA_budget3_ratio5_seed42.json", "loveda_budget3_topcopy", "LoveDA"),
        ("results/loveda/LoveDA_scsr_5_seed42.json", "loveda_scsr_topcopy", "LoveDA"),
        ("results/loveda/LoveDA_scsr_task_5_seed42.json", "loveda_scsr_task_topcopy", "LoveDA"),
        ("results/loveda/LoveDA_scsr_v2_5_seed42.json", "loveda_scsr_v2_topcopy", "LoveDA"),
        ("results/isaid/iSAID/iSAID_full.json", "isaid_full", "iSAID"),
    ]
    for rel, variant, ds in singles:
        p = CLOUD / rel
        if p.exists():
            try:
                rows.append(add_single_run_file(p, variant, ds))
            except Exception as e:  # noqa: BLE001
                print(f"WARN: failed to parse {p}: {e}")

    # 3. multiseed aggregates
    rows.extend(aggregate_rows_from_multiseed())

    # sort: dataset -> ratio (numeric) -> suite -> seed
    def sort_key(r):
        def num(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return 1e9

        return (
            r.get("dataset", ""),
            num(r.get("ratio")),
            r.get("suite", ""),
            r.get("seed", ""),
        )

    rows.sort(key=sort_key)

    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[c[0] for c in COLUMNS])
        writer.writeheader()
        for r in rows:
            writer.writerow({c[0]: r.get(c[0], "") for c in COLUMNS})

    n_run = sum(1 for r in rows if r.get("record_type") == "run")
    n_agg = sum(1 for r in rows if r.get("record_type") == "aggregate")
    print(f"Wrote {OUT}")
    print(f"  total rows: {len(rows)}  (runs={n_run}, aggregates={n_agg})")


if __name__ == "__main__":
    main()
