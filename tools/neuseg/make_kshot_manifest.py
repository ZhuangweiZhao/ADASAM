"""Create reproducible same-class K-shot manifests for NEU_Seg."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adasam.datasets import NEUSegDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/NEU_Seg")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dataset = NEUSegDataset(args.data_root, split="train")
    rng = random.Random(args.seed)
    selected = set()
    class_ids = {str(class_id): [] for class_id in (1, 2, 3)}
    for class_id in (1, 2, 3):
        candidates = [
            index for index in range(len(dataset))
            if (dataset[index]["masks"] == class_id).any()
        ]
        rng.shuffle(candidates)
        chosen = candidates if args.k <= 0 or args.k >= len(candidates) else candidates[: args.k]
        class_ids[str(class_id)] = [dataset.sample_names[index] for index in chosen]
        selected.update(chosen)
    manifest = {
        "k_shot": args.k if args.k > 0 else len(selected),
        "seed": args.seed,
        "sample_ids": [dataset.sample_names[index] for index in sorted(selected)],
        "class_sample_ids": class_ids,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "k": manifest["k_shot"], "samples": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
