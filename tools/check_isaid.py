"""Check completeness of the official iSAID semantic dataset layout."""

from __future__ import annotations

import argparse
from pathlib import Path

from adasam.datasets.industrial import ISAIDSemanticDataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    args = p.parse_args()
    root = Path(args.data_root)
    print(f"root={root}")
    for split in ("train", "val", "test"):
        try:
            dataset = ISAIDSemanticDataset(root, split, 800)
            print(f"{split}: samples={len(dataset)}")
            if split != "test":
                sample = dataset[0]
                print(f"  image={tuple(sample['image'].shape)} mask={tuple(sample['mask'].shape)} labels={sorted(sample['mask'].unique().tolist())}")
        except Exception as exc:
            print(f"{split}: ERROR: {exc}")
    print("expected train images: TrainData/train/images/images/*.png")
    print("expected val images:   ValidationData/val/images/images/*.png")
    print("expected test images:  TestData/TestData/Part1-001/images/*.png and Part1-002/images/*.png")


if __name__ == "__main__":
    main()
