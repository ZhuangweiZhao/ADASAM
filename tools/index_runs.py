"""Create an inventory of the organized experiment run tree."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CATEGORIES = ("current", "development", "history", "sanity", "diagnostics", "empty")


def experiment_directories() -> list[Path]:
    directories = []
    markers = {
        "metrics.json", "evaluation.json", "survival_summary.json",
        "efficiency_summary.json", "summary.json",
    }
    for directory in RUNS.rglob("*"):
        if not directory.is_dir():
            continue
        names = {path.name for path in directory.iterdir() if path.is_file()}
        if names & markers or any(path.suffix.lower() == ".pt" for path in directory.iterdir() if path.is_file()):
            directories.append(directory)
    empty_root = RUNS / "empty"
    if empty_root.exists():
        directories.extend(path for path in empty_root.rglob("*") if path.is_dir())
    return sorted(set(directories), key=lambda path: str(path.relative_to(RUNS)))


def main() -> None:
    rows = []
    for directory in experiment_directories():
        relative = directory.relative_to(RUNS)
        files = [path for path in directory.iterdir() if path.is_file()]
        category = relative.parts[0]
        rows.append({
            "path": str(relative).replace("\\", "/"),
            "category": category,
            "status": "active" if category == "current" else "historical",
            "files_here": len(files),
            "checkpoints_here": sum(path.suffix.lower() == ".pt" for path in files),
            "metrics_here": sum(path.name in {"metrics.json", "evaluation.json"} for path in files),
            "size_mb_here": round(sum(path.stat().st_size for path in files) / 1024**2, 2),
            "last_modified": datetime.fromtimestamp(directory.stat().st_mtime).isoformat(timespec="seconds"),
        })

    fields = list(rows[0]) if rows else []
    with (RUNS / "run_index.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (RUNS / "run_index.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "categories": list(CATEGORIES),
                "runs": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"indexed {len(rows)} experiment directories")


if __name__ == "__main__":
    main()
