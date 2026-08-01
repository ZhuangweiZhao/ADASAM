"""Run and summarize the Phase 1 NEU_Seg label-ratio benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the label-ratio benchmark")
    parser.add_argument("--ratios", nargs="+", type=int, default=[1, 5, 10, 25, 100])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--data-root", default="data/NEU_Seg")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="runs/label_ratio_benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = _REPO_ROOT / output_root
    for ratio in args.ratios:
        command = [
            sys.executable,
            str(_REPO_ROOT / "tools" / "train_segmentation.py"),
            "--dataset", "neu_seg",
            "--label_ratio", str(ratio),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--data-root", args.data_root,
            "--device", args.device,
            "--seed", str(args.seed),
            "--output-dir", str(output_root),
        ]
        subprocess.run(command, cwd=_REPO_ROOT, check=True)

    rows = []
    for ratio in args.ratios:
        path = output_root / f"neu_seg_ratio{ratio}_seed{args.seed}" / "metrics.json"
        metrics = json.loads(path.read_text(encoding="utf-8"))
        test = metrics["test"]
        rows.append(
            {
                "ratio": ratio,
                "labeled_images": metrics["label_pool_samples"],
                "mIoU": test["mIoU"],
                "mIoU_fg": test["mIoU_fg"],
                "Dice": test["Dice"],
                "Dice_fg": test["Dice_fg"],
                "train_time_seconds": sum(item["seconds"] for item in metrics["history"]),
                "FPS": test["FPS"],
                "best_epoch": metrics["best_epoch"],
            }
        )
    summary = {
        "protocol": {
            "dataset": "NEU_Seg",
            "seed": args.seed,
            "epochs": args.epochs,
            "validation_fraction": 0.2,
            "test_images": 840,
        },
        "results": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
