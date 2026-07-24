"""
Download SAM ViT-H checkpoint for SAM-RSP Stage 3.
==================================================

Downloads the Segment Anything ViT-H encoder weights (~2.4 GB).
Required for SAM-RSP Stage 3 training.

用法 | Usage::

    python tools/download_sam_weight.py

输出 | Output::

    weights/sam_vit_h_4b8939.pth  (~2.4 GB)
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = _REPO_ROOT / "weights" / "sam_vit_h_4b8939.pth"
URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT.exists():
        size_gb = OUTPUT.stat().st_size / 1e9
        print(f"SAM checkpoint already exists: {OUTPUT} ({size_gb:.1f} GB)")
        print("Delete it first to re-download.")
        return

    print(f"Downloading SAM ViT-H weights (~2.4 GB)...")
    print(f"URL: {URL}")
    print(f"Output: {OUTPUT}")
    print(f"Please be patient, this will take a while...")
    print()

    try:
        import urllib.request
        urllib.request.urlretrieve(URL, str(OUTPUT))
    except ImportError:
        # Python without urllib — use wget/curl
        import subprocess
        print("urllib not available, trying wget...")
        result = subprocess.run(
            ["wget", "-O", str(OUTPUT), URL],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print("wget failed. Please download manually:")
            print(f"  {URL}")
            print(f"  → save to {OUTPUT}")
            sys.exit(1)

    size_gb = OUTPUT.stat().st_size / 1e9
    print(f"\nDownload complete: {size_gb:.1f} GB")
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
