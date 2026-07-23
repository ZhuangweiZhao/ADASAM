"""
NEU_Seg 工业缺陷分割数据集 | Industrial Defect Segmentation Dataset.
====================================================================

200×200 表面缺陷图块, 4 类 (含背景)。适配 AdaSAM 小样本训练与评估。
200×200 surface defect tiles, 4 classes (incl. background). Fits AdaSAM few-shot training.

类别编码 | Class Encoding:
    0 = Background (背景)
    1 = Inclusion (夹杂物)
    2 = Patch (斑块)
    3 = Scratch (划痕)

目录结构 | Directory Structure:
    NEU_Seg/
    ├── images/
    │   ├── training/       # 3630 JPG images
    │   └── test/           # 840 JPG images
    └── annotations/
        ├── training/       # 3630 PNG masks (pixel value 0/1/2/3)
        └── test/           # 840 PNG masks

样本契约 | Sample contract::

    {
        "image":      Tensor[3, H, W] float32, RGB, ∈ [0, 1],
        "masks":      Tensor[1, H, W] int64, class labels {0,1,2,3},
        "image_id":   str,           # 样本名 (无扩展名) | sample name (no extension)
        "image_size": (H, W),        # 原始图像尺寸 (200, 200)
    }

用法 | Usage::

    from adasam.datasets import NEUSegDataset

    ds = NEUSegDataset(root="data/NEU_Seg", split="train")
    sample = ds[0]
    # sample["image"]: [3, 200, 200] float32
    # sample["masks"]: [1, 200, 200] int64  {0,1,2,3}
    # sample["image_id"]: "000201"
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from adasam.logging import get_logger

logger = get_logger("dataset.neuseg")

# ── 类别常量 | Class constants ──
NUM_CLASSES = 4  # 0=BG, 1=Inclusion, 2=Patch, 3=Scratch
CLASS_NAMES = ["background", "Inclusion", "Patch", "Scratch"]
CLASS_COLORS = {
    0: (128, 128, 128),  # BG: gray
    1: (255, 0, 0),      # Inclusion: red
    2: (0, 255, 0),      # Patch: green
    3: (0, 0, 255),      # Scratch: blue
}
NEUSEG_CATEGORIES: dict[int, str] = {i: n for i, n in enumerate(CLASS_NAMES)}
NEUSEG_CLASS_ID = 1
NEUSEG_CLASS_NAME = "defect"

_SPLIT_MAP = {"train": "training", "val": "test", "test": "test"}


class NEUSegDataset(Dataset):
    """NEU_Seg 工业缺陷分割数据集 | Industrial Defect Segmentation Dataset.

    200×200 表面缺陷图块, 4 类别 (含背景)。
    200×200 surface defect tiles, 4 classes (incl. background).

    :param root: 数据根目录 | data root (e.g. "data/NEU_Seg").
    :param split: "train" → images/training/, "val" or "test" → images/test/.
    :param transforms: 可选数据增强 | optional data augmentation.
    """

    NUM_CLASSES = NUM_CLASSES
    CLASS_NAMES = CLASS_NAMES

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transforms=None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train'/'val'/'test', got '{split}'")

        super().__init__()
        self._root = Path(root)
        self.split = split
        self.transforms = transforms

        dir_name = _SPLIT_MAP.get(split, split)
        self._img_dir = self._root / "images" / dir_name
        self._ann_dir = self._root / "annotations" / dir_name

        for d, name in [(self._img_dir, "images"), (self._ann_dir, "annotations")]:
            if not d.exists():
                raise FileNotFoundError(
                    f"目录未找到 | Directory not found: {d}\n"
                    f"Expected: root/images/{dir_name}/ and root/annotations/{dir_name}/"
                )

        # ── 扫描文件 | Scan files ──
        img_files = sorted([f.stem for f in self._img_dir.glob("*.jpg")])
        self._samples = [
            name for name in img_files
            if (self._ann_dir / f"{name}.png").exists()
        ]

        skipped = len(img_files) - len(self._samples)
        if skipped > 0:
            logger.log_warn(
                "dataset/missing_masks",
                f"跳过 {skipped} 张无标注图像 | Skipped {skipped} images without mask in {dir_name}",
            )

        logger.log_info(
            "dataset/neuseg_init",
            f"NEU_Seg ({dir_name}): {len(self)} samples, "
            f"num_classes={NUM_CLASSES}, image_size=200x200",
        )

    # ── 核心接口 | Core interface ──

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict:
        name = self._samples[index]

        # 加载图像 | Load image
        img = cv2.imread(str(self._img_dir / f"{name}.jpg"), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图像 | Cannot read image: {self._img_dir / f'{name}.jpg'}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = torch.from_numpy(img).permute(2, 0, 1).float()  # [3, 200, 200]

        # 加载标注 | Load annotation (0/1/2/3)
        ann = cv2.imread(str(self._ann_dir / f"{name}.png"), cv2.IMREAD_UNCHANGED)
        if ann is None:
            raise ValueError(f"无法读取标注 | Cannot read annotation: {self._ann_dir / f'{name}.png'}")
        masks = torch.from_numpy(ann.astype(np.int64)).unsqueeze(0)  # [1, 200, 200]

        sample = {
            "image": image,
            "masks": masks,
            "image_id": name,
            "image_size": tuple(image.shape[1:]),  # (200, 200)
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    # ── 查询接口 | Query interface ──

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES

    @property
    def class_names(self) -> list[str]:
        return list(CLASS_NAMES)

    @property
    def sample_names(self) -> list[str]:
        return list(self._samples)

    @property
    def image_size(self) -> tuple[int, int]:
        return (200, 200)

    def get_class_pixel_counts(self) -> dict[str, int]:
        """统计各类别像素数 | Count pixels per class (lazy)."""
        if not hasattr(self, "_pixel_counts"):
            counts = {c: 0 for c in range(NUM_CLASSES)}
            for i in range(len(self)):
                ann = cv2.imread(str(self._ann_dir / f"{self._samples[i]}.png"), cv2.IMREAD_UNCHANGED)
                if ann is not None:
                    for c in range(NUM_CLASSES):
                        counts[c] += int((ann == c).sum())
            self._pixel_counts = counts
        return {CLASS_NAMES[c]: self._pixel_counts[c] for c in range(NUM_CLASSES)}

    def get_fg_ratio(self, class_id: int | None = None) -> float:
        """前景占比 | FG ratio (None=all FG classes)."""
        counts = self.get_class_pixel_counts()
        total = sum(counts.values())
        if class_id is not None:
            return counts.get(CLASS_NAMES[class_id], 0) / max(total, 1)
        fg = sum(v for k, v in counts.items() if "background" not in k.lower())
        return fg / max(total, 1)
