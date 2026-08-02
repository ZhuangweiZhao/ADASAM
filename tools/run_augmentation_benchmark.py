"""Run the isolated NEU-Seg augmentation ablation without changing old experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run defect-aware augmentation ablations")
    parser.add_argument("--ratios", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-root", default="data/NEU_Seg")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="runs/augmentation_benchmark")
    parser.add_argument(
        "--full-matrix",
        action="store_true",
        help="run none/basic/defect for both baseline and DAPG-v2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = _REPO_ROOT / output_root
    experiments = (
        [("baseline", "none"), ("baseline", "basic"), ("baseline", "defect"),
         ("dapg_v2", "none"), ("dapg_v2", "basic"), ("dapg_v2", "defect")]
        if args.full_matrix
        else [("baseline", "basic"), ("baseline", "defect")]
    )
    rows = []
    for model_name, augmentation in experiments:
        experiment_dir = output_root / f"{model_name}_{augmentation}"
        command = [
            sys.executable,
            str(_REPO_ROOT / "tools" / "run_label_ratio_benchmark.py"),
            "--ratios", *[str(ratio) for ratio in args.ratios],
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--data-root", args.data_root,
            "--adapter", "cat",
            "--augmentation", augmentation,
            "--device", args.device,
            "--seed", str(args.seed),
            "--output-dir", str(experiment_dir),
        ]
        if model_name == "dapg_v2":
            command.extend(["--prompt-version", "v2", "--prompt-fusion-mode", "both"])
        subprocess.run(command, cwd=_REPO_ROOT, check=True)
        summary = json.loads((experiment_dir / "summary.json").read_text(encoding="utf-8"))
        for result in summary["results"]:
            rows.append({"experiment": model_name, "augmentation": augmentation, **result})
    output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": {
            "dataset": "NEU_Seg",
            "seed": args.seed,
            "epochs": args.epochs,
            "ratios": args.ratios,
            "validation_augmentation": "none",
            "test_augmentation": "none",
        },
        "results": rows,
    }
    (output_root / "augmentation_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
