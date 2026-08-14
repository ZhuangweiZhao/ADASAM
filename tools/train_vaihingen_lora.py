"""Convenience entry point for the Vaihingen MobileSAM LoRA experiment."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.train_vaihingen import main


if __name__ == "__main__":
    if "--lora-rank" not in sys.argv:
        sys.argv.extend(["--lora-rank", "4", "--lora-alpha", "8"])
    main()
