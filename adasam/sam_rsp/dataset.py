"""
SAM-RSP iSAID-5i 数据集适配器 | iSAID-5i Dataset Adapter for SAM-RSP.
======================================================================

为 SAM-RSP 三阶段训练提供 iSAID-5i 数据加载:

  Stage 1 (PSPNet): 多类语义分割 (base classes + BG)
  Stage 2 (BAM):    二类 few-shot episodes (FG/BG, base classes)
  Stage 3 (SAM-RSP): 二类 few-shot episodes (FG/BG, base classes)
  Test:              二类 few-shot episodes (FG/BG, novel classes)

数据来源: tools/sam_rsp_prepare_isaid.py 生成的 sam_rsp/lists/foldX/*.txt
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# iSAID-5i class info (same as adasam.datasets.isaid_5i)
NUM_CLASSES = 15
CLASS_NAMES = [
    "BG", "ship", "storage_tank", "baseball_diamond", "tennis_court",
    "basketball_court", "ground_track_field", "bridge", "large_vehicle",
    "small_vehicle", "helicopter", "swimming_pool", "roundabout",
    "soccer_ball_field", "plane", "harbor",
]
ISAID5I_FOLDS: dict[int, dict[str, list[int]]] = {
    0: {"test": [1, 2, 3, 4, 5],       "train": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]},
    1: {"test": [6, 7, 8, 9, 10],       "train": [1, 2, 3, 4, 5, 11, 12, 13, 14, 15]},
    2: {"test": [11, 12, 13, 14, 15],   "train": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
}

# Standard ImageNet mean/std for normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0


# ═══════════════════════════════════════════════════════════════════
# Stage 1: PSPNet Dataset (multi-class semantic segmentation)
# ═══════════════════════════════════════════════════════════════════

class PSPNetDataset(Dataset):
    """Stage 1 数据集: 多类语义分割 (base classes remapped to 1..N).

    Multi-class semantic segmentation on base classes.
    BG=0, base class IDs remapped to contiguous 1..N.
    """

    def __init__(
        self,
        data_root: str | Path,
        fold: int = 0,
        transform: Callable | None = None,
    ):
        self.data_root = Path(data_root)
        self.fold = fold
        self.transform = transform

        fold_def = ISAID5I_FOLDS[fold]
        self.base_classes = sorted(fold_def["train"])  # [6-15] for fold 0
        self._cls_to_idx = {c: i + 1 for i, c in enumerate(self.base_classes)}  # 6→1, 7→2, ...

        # Load data list
        list_file = self.data_root / "sam_rsp" / "lists" / f"fold{fold}" / "base_train.txt"
        self._items: list[tuple[str, str]] = []
        with open(list_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    img_p, ann_p = line.split(" ", 1)
                    self._items.append((img_p, ann_p.strip()))

        print(f"[PSPNetDataset] fold={fold} base_classes={self.base_classes} "
              f"num_classes={len(self.base_classes)} items={len(self._items)}")

    @property
    def num_classes(self) -> int:
        return len(self.base_classes)  # excluding BG

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict:
        img_path, ann_path = self._items[index]

        # Load image
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Load mask (grayscale, values are original class IDs)
        mask = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE).astype(np.int64)

        # Remap: original class ID → contiguous index (0=BG, 1..N=base classes)
        remapped = np.zeros_like(mask, dtype=np.int64)
        for orig_cls, new_idx in self._cls_to_idx.items():
            remapped[mask == orig_cls] = new_idx

        if self.transform:
            image, remapped = self.transform(image, remapped)

        return {
            "image": torch.from_numpy(image).permute(2, 0, 1).float(),  # [3, H, W]
            "mask": torch.from_numpy(remapped).long(),                   # [H, W]
        }


# ═══════════════════════════════════════════════════════════════════
# Stage 2/3 & Test: Few-shot Episode Dataset (FG/BG binary)
# ═══════════════════════════════════════════════════════════════════

class FewShotEpisodeDataset(Dataset):
    """Stage 2/3 和测试数据集: 二类 few-shot episodes.

    Binary FG/BG few-shot episodes for BAM, SAM-RSP training and testing.
    每张 query tile 随机选择一个可见类别, 构建:
      - query image + binary query mask (该类像素=1)
      - K support images + binary support masks
    """

    def __init__(
        self,
        data_root: str | Path,
        fold: int = 0,
        shot: int = 1,
        use_base: bool = True,  # True=base classes (train), False=novel (test)
        split: str = "train",   # "train" or "val"
        transform: Callable | None = None,
        seed: int = 42,
    ):
        self.data_root = Path(data_root)
        self.fold = fold
        self.shot = shot
        self.use_base = use_base
        self.split = split
        self.transform = transform

        fold_def = ISAID5I_FOLDS[fold]
        if use_base:
            self.visible_classes = sorted(fold_def["train"])
            prefix = "base" if split == "train" else "base_val"
            # base_train is the only base data
            list_name = "base_train"
        else:
            self.visible_classes = sorted(fold_def["test"])
            list_name = f"novel_{split}"  # novel_train or novel_val

        # Load class→tile mapping
        base = self.data_root / "sam_rsp" / "lists" / f"fold{fold}"
        classes_file = base / f"{list_name}_classes.txt"
        data_list_file = base / f"{list_name}.txt"

        # Load data_list (all tiles)
        self._data_list: list[tuple[str, str]] = []
        if data_list_file.exists():
            with open(data_list_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        img_p, ann_p = line.split(" ", 1)
                        self._data_list.append((img_p, ann_p.strip()))

        # Load class→tile mapping
        self._class_tiles: dict[int, list[tuple[str, str]]] = defaultdict(list)
        if classes_file.exists():
            with open(classes_file, encoding="utf-8") as f:
                raw = f.read()
                # eval is safe here since we generated the file ourselves
                class_dict = eval(raw)
                for cls_str, items in class_dict.items():
                    cls_id = int(cls_str)
                    if cls_id in self.visible_classes:
                        self._class_tiles[cls_id] = items

        self._rng = random.Random(seed)

        visible_str = ",".join(
            f"{c}({CLASS_NAMES[c]})" for c in self.visible_classes
        )
        n_tiles = len(self._data_list)
        n_pairs = sum(len(v) for v in self._class_tiles.values())
        print(f"[FewShotEpisode] fold={fold} {'base' if use_base else 'novel'}"
              f" {split} shot={shot}: {n_tiles} tiles, {n_pairs} class-tile pairs")
        print(f"  visible: {visible_str}")

        if n_tiles == 0:
            raise RuntimeError(
                f"No tiles found for fold={fold} {'base' if use_base else 'novel'} {split}.\n"
                f"Expected data at: {base}\n"
                "Run the data preparation script first:\n"
                f"  python tools/sam_rsp_prepare_isaid.py --data-root {self.data_root}"
            )

    def __len__(self) -> int:
        return len(self._data_list)

    def __getitem__(self, index: int) -> dict:
        # --- Query ---
        q_img_path, q_ann_path = self._data_list[index]

        # Load query image
        q_image = cv2.imread(q_img_path, cv2.IMREAD_COLOR)
        q_image = cv2.cvtColor(q_image, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Load query mask → find visible classes present
        q_mask = cv2.imread(q_ann_path, cv2.IMREAD_GRAYSCALE)
        mask_classes = sorted({
            int(c) for c in np.unique(q_mask) if c in self.visible_classes
        })
        if not mask_classes:
            # Return a random class from visible if none present in this tile
            chosen_cls = self._rng.choice(self.visible_classes)
        else:
            chosen_cls = self._rng.choice(mask_classes)

        # Create binary query mask (chosen_cls = 1, rest = 0)
        q_binary = (q_mask == chosen_cls).astype(np.float32)

        # --- Support ---
        support_tiles = self._class_tiles.get(chosen_cls, [])
        if len(support_tiles) < self.shot:
            # Not enough tiles for this class → use what we have, with replacement
            support_indices = self._rng.choices(
                range(len(support_tiles)), k=self.shot
            ) if support_tiles else [0] * self.shot
        else:
            support_indices = self._rng.sample(range(len(support_tiles)), self.shot)

        s_images = []
        s_masks = []
        for si in support_indices:
            if si < len(support_tiles):
                s_img_p, s_ann_p = support_tiles[si]
            else:
                s_img_p, s_ann_p = q_img_path, q_ann_path  # fallback to query

            s_img = cv2.imread(s_img_p, cv2.IMREAD_COLOR)
            s_img = cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB).astype(np.float32)
            s_mask_raw = cv2.imread(s_ann_p, cv2.IMREAD_GRAYSCALE)
            s_binary = (s_mask_raw == chosen_cls).astype(np.float32)

            s_images.append(s_img)
            s_masks.append(s_binary)

        # --- Transform ---
        if self.transform:
            q_image, q_binary = self.transform(q_image, q_binary)
            for k in range(self.shot):
                s_images[k], s_masks[k] = self.transform(s_images[k], s_masks[k])

        # Convert to tensors
        q_image_t = torch.from_numpy(q_image).permute(2, 0, 1).float()  # [3, H, W]
        q_mask_t = torch.from_numpy(q_binary).long()                     # [H, W]

        s_images_t = torch.stack([
            torch.from_numpy(si).permute(2, 0, 1).float() for si in s_images
        ], dim=0)  # [K, 3, H, W]
        s_masks_t = torch.stack([
            torch.from_numpy(sm).long() for sm in s_masks
        ], dim=0)  # [K, H, W]

        # Map class to sub-index (0-based, for SAM-RSP's subcls)
        subcls = self.visible_classes.index(chosen_cls) if chosen_cls in self.visible_classes else 0

        return {
            "query_image": q_image_t,
            "query_mask": q_mask_t,
            "support_images": s_images_t,
            "support_masks": s_masks_t,
            "class_id": chosen_cls,
            "subcls": subcls,
        }
