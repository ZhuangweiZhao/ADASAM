"""Run the standardized LoveDA PEFT screening matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", default=str(ROOT / "weights" / "mobile_sam.pt"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--methods", nargs="+", choices=["decoder", "cat", "lora", "full_ft"],
        default=["decoder", "cat", "lora", "full_ft"],
    )
    parser.add_argument("--ratios", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--rerun-completed", action="store_true",
        help="run experiments even when their metrics.json already exists",
    )
    return parser.parse_args()


def command_for(args: argparse.Namespace, method: str, ratio: int, seed: int) -> list[str]:
    model = "mobilesam_finetune" if method == "full_ft" else "mobilesam"
    command = [
        sys.executable, str(ROOT / "tools" / "train_loveda.py"),
        "--model", model,
        "--data-root", args.data_root,
        "--checkpoint", args.checkpoint,
        "--label-ratio", str(ratio),
        "--image-size", "512",
        "--sam-image-size", "512",
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--epochs", str(args.epochs),
        "--augmentation", "remote_strong",
        "--rural-sampling-multiplier", "1.5",
        "--adapter", "cat" if method == "cat" else "none",
        "--adapter-placement", "pre_fusion",
        "--feature-scales", "p3_p4_embedding",
        "--fusion-version", "sum",
        "--decoder-version", "lightweight",
        "--class-balanced-ce",
        "--lovasz-weight", "0.5",
        "--lr", "0.0003",
        "--weight-decay", "0.0001",
        "--lr-scheduler", "cosine",
        "--grad-clip-norm", "1.0",
        "--validation-seed", "42",
        "--seed", str(seed),
        "--device", args.device,
        "--output-dir", str(Path(args.output_dir) / method),
    ]
    if method == "lora":
        command.extend(["--lora-rank", "4", "--lora-alpha", "8", "--lora-targets", "qkv", "proj"])
    if method == "full_ft":
        command.extend(["--backbone-lr-multiplier", "0.1"])
    return command


def expected_metrics_path(args: argparse.Namespace, method: str, ratio: int, seed: int) -> Path:
    if method == "lora":
        run_name = f"mobilesam_lora_r4_qkv-proj_ratio{ratio}_seed{seed}"
    elif method == "full_ft":
        run_name = f"mobilesam_finetune_ratio{ratio}_seed{seed}"
    else:
        run_name = f"loveda_ratio{ratio}_seed{seed}"
    return Path(args.output_dir) / method / run_name / "metrics.json"


def main() -> None:
    args = parse_args()
    invalid_ratios = sorted(set(args.ratios) - {1, 5, 10, 20, 25, 50, 100})
    if invalid_ratios:
        raise ValueError(f"unsupported label ratios: {invalid_ratios}")
    for method in args.methods:
        for ratio in args.ratios:
            for seed in args.seeds:
                command = command_for(args, method, ratio, seed)
                metrics_path = expected_metrics_path(args, method, ratio, seed)
                if metrics_path.exists() and not args.rerun_completed:
                    print(f"SKIP completed={metrics_path}", flush=True)
                    continue
                print("COMMAND", subprocess.list2cmdline(command), flush=True)
                if not args.dry_run:
                    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
