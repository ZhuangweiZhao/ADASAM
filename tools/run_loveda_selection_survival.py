"""Run the fixed LoRA+Sum 5% sample-selection survival experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("random", "embedding_kcenter", "hrcs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", default=str(ROOT / "weights" / "mobile_sam.pt"))
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def command_for(args: argparse.Namespace, method: str) -> list[str]:
    manifest = Path(args.manifest_dir) / f"{method}_ratio5.json"
    return [
        sys.executable, str(ROOT / "tools" / "train_loveda.py"),
        "--model", "mobilesam",
        "--data-root", args.data_root,
        "--checkpoint", args.checkpoint,
        "--label-ratio", "5",
        "--selection-manifest", str(manifest),
        "--image-size", "512",
        "--sam-image-size", "512",
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--epochs", str(args.epochs),
        "--augmentation", "remote_strong",
        "--rural-sampling-multiplier", "1.5",
        "--adapter", "none",
        "--lora-rank", "4",
        "--lora-alpha", "8",
        "--lora-targets", "qkv", "proj",
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
        "--seed", "42",
        "--device", args.device,
        "--output-dir", str(Path(args.output_dir) / method),
    ]


def main() -> None:
    args = parse_args()
    for method in args.methods:
        command = command_for(args, method)
        manifest = Path(args.manifest_dir) / f"{method}_ratio5.json"
        if not args.dry_run and not manifest.exists():
            raise FileNotFoundError(f"selection manifest not found: {manifest}")
        print("COMMAND", subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
