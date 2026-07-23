"""
iSAID-5i 小样本语义分割数据集 | Few-shot Semantic Segmentation Dataset.
======================================================================

iSAID-5i 标准小样本协议: 15 类, 3-fold 交叉验证 (每 fold 5 测试类 + 10 训练类),
256×256 航拍图块, 类级语义标注。

Standard iSAID-5i few-shot protocol: 15 classes, 3-fold cross-validation
(5 test + 10 train classes per fold), 256×256 aerial tiles, class-level labels.

目录结构 | Directory Structure:
    iSAID-5i/iSAID/
    ├── train/
    │   ├── images/           # 256×256 RGB PNG tiles
    │   ├── semantic_mask/    # RGB color-coded masks
    │   ├── semantic_png/     # Grayscale class labels (0=BG, 1-15)
    │   ├── instance_mask/    # RGB instance-id masks
    │   └── train_list/       # split0/1/2_train.txt
    └── val/
        ├── images/
        ├── semantic_mask/
        ├── semantic_png/
        ├── instance_mask/
        └── val_list/         # split0/1/2_val.txt

样本契约 | Sample contract::

    {
        "image":      Tensor[3, H, W] float32, RGB, ∈ [0, 1],
        "instances":  list[{"category_id": int, "mask": Tensor[H,W] bool}],
        "image_id":   int,           # 内部索引 | internal index
        "image_size": (H, W),        # 原始尺寸 (256, 256)
        "classes":    set[int],      # 该 tile 中的类别 | classes present
    }

用法 | Usage::

    from adasam.datasets.isaid_5i import ISAID5iDataset, ISAID5iEpisodeSampler

    ds = ISAID5iDataset(root="data/iSAID-5i", fold=0, split="train")
    sampler = ISAID5iEpisodeSampler(ds, k_shot=5, seed=42)
    episode = sampler.sample()
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from adasam.logging import get_logger

logger = get_logger("dataset.isaid5i")

# ── iSAID-5i 类别 | Class constants ──
NUM_CLASSES = 15  # foreground classes only (1-15), 0=BG
MIN_INSTANCE_AREA = 16  # 小于此像素面积的实例丢弃 | drop instances below this pixel area
CLASS_NAMES = [
    "BG", "ship", "storage_tank", "baseball_diamond", "tennis_court",
    "basketball_court", "ground_track_field", "bridge", "large_vehicle",
    "small_vehicle", "helicopter", "swimming_pool", "roundabout",
    "soccer_ball_field", "plane", "harbor",
]

# Standard 5i folds (class IDs 1-15):
# Fold 0: test={1,2,3,4,5}, train={6,7,8,9,10,11,12,13,14,15}
# Fold 1: test={6,7,8,9,10}, train={1,2,3,4,5,11,12,13,14,15}
# Fold 2: test={11,12,13,14,15}, train={1,2,3,4,5,6,7,8,9,10}
ISAID5I_FOLDS: dict[int, dict[str, list[int]]] = {
    0: {"test": [1, 2, 3, 4, 5],       "train": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]},
    1: {"test": [6, 7, 8, 9, 10],       "train": [1, 2, 3, 4, 5, 11, 12, 13, 14, 15]},
    2: {"test": [11, 12, 13, 14, 15],   "train": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
}

ISAID5I_CATEGORIES: dict[int, str] = {i: CLASS_NAMES[i] for i in range(1, 16)}


# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════

class ISAID5iDataset(Dataset):
    """iSAID-5i 小样本语义分割数据集 | Few-shot Semantic Segmentation Dataset.

    :param root: 数据根目录 | data root (e.g. "data/iSAID-5i").
    :param fold: fold 编号 0/1/2 | fold index.
    :param split: "train" 或 "val" | "train" or "val".
    :param mode: "base" | "novel" | "all" — 控制可见类别集合.
        "base" = 使用 fold 的训练类, "novel" = 使用测试类, "all" = 全部 15 类.
    """

    NUM_CLASSES = NUM_CLASSES
    CLASS_NAMES = CLASS_NAMES

    def __init__(
        self,
        root: str | Path,
        fold: int = 0,
        split: str = "train",
        mode: str = "novel",
    ) -> None:
        if fold not in (0, 1, 2):
            raise ValueError(f"fold must be 0/1/2, got {fold}")
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train'/'val', got '{split}'")
        if mode not in ("base", "novel", "all"):
            raise ValueError(f"mode must be 'base'/'novel'/'all', got '{mode}'")

        super().__init__()
        self.root = Path(root) / "iSAID"
        self.fold = fold
        self.split = split
        self.mode = mode

        # Determine visible classes
        fold_def = ISAID5I_FOLDS[fold]
        if mode == "base":
            self._visible_classes = fold_def["train"]
        elif mode == "novel":
            self._visible_classes = fold_def["test"]
        else:
            self._visible_classes = list(range(1, 16))

        # Directory setup
        if split == "train":
            self._img_dir = self.root / "train" / "images"
            self._ann_dir = self.root / "train" / "semantic_png"
            list_file = self.root / "train" / "train_list" / f"split{fold}_train.txt"
        else:
            self._img_dir = self.root / "val" / "images"
            self._ann_dir = self.root / "val" / "semantic_png"
            list_file = self.root / "val" / "val_list" / f"split{fold}_val.txt"

        for d, name in [(self._img_dir, "images"), (self._ann_dir, "semantic_png")]:
            if not d.exists():
                raise FileNotFoundError(f"{name} directory not found: {d}")

        # Parse split file to get tile→class mapping
        self._tile_classes: dict[str, set[int]] = defaultdict(set)
        self._source_images: dict[str, str] = {}  # tile_id → source image name

        if list_file.exists():
            with open(list_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Format: P1092_1648_1904_824_1080_instance_color_RGB.png_04
                    parts = line.rsplit("_", 1)
                    rest = parts[0]
                    tile_id = rest.replace("_instance_color_RGB.png", "")
                    # Extract source image: P{id}_{coords} → P{id}
                    src = tile_id.split("_")[0]
                    self._tile_classes[tile_id]  # ensure key exists
                    self._source_images[tile_id] = src

        # Build tile index: filter tiles that have visible classes
        self._tiles: list[str] = []
        self._tile_class_map: dict[str, list[int]] = {}  # tile_id → [class_ids visible]

        for tile_id in sorted(self._tile_classes.keys()):
            img_path = self._img_dir / f"{tile_id}.png"
            ann_path = self._ann_dir / f"{tile_id}_instance_color_RGB.png"
            if not img_path.exists() or not ann_path.exists():
                continue

            # Check which visible classes are present in this tile
            ann = cv2.imread(str(ann_path), cv2.IMREAD_UNCHANGED)
            if ann is None:
                continue
            present = {int(c) for c in np.unique(ann)} & set(self._visible_classes)
            if present:
                self._tiles.append(tile_id)
                self._tile_class_map[tile_id] = sorted(present)

        # Build class→tiles index
        self._class_tiles: dict[int, list[int]] = defaultdict(list)
        for idx, tile_id in enumerate(self._tiles):
            for cls in self._tile_class_map[tile_id]:
                if cls in self._visible_classes:
                    self._class_tiles[cls].append(idx)

        n_tiles = len(self._tiles)
        n_class_tiles = sum(len(v) for v in self._class_tiles.values())
        logger.log_info(
            "dataset/isaid5i_init",
            f"iSAID-5i fold={fold} {split}/{mode}: {n_tiles} tiles, "
            f"{n_class_tiles} class-tile pairs, {len(self.visible_classes())} classes "
            f"({[ISAID5I_CATEGORIES.get(c, str(c)) for c in sorted(self.visible_classes())]})",
        )

    # ── 核心接口 | Core interface ──

    def __len__(self) -> int:
        return len(self._tiles)

    def __getitem__(self, index: int) -> dict:
        tile_id = self._tiles[index]
        H = W = 256

        # Load image
        img = cv2.imread(str(self._img_dir / f"{tile_id}.png"), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {self._img_dir / f'{tile_id}.png'}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = torch.from_numpy(img).permute(2, 0, 1).float()  # [3, 256, 256]

        # Load annotation (grayscale class labels 0-15)
        ann = cv2.imread(
            str(self._ann_dir / f"{tile_id}_instance_color_RGB.png"), cv2.IMREAD_UNCHANGED
        )
        if ann is None:
            raise ValueError(
                f"Cannot read annotation: {self._ann_dir / f'{tile_id}_instance_color_RGB.png'}"
            )

        # Build per-class instance list — split connected components
        instances = []
        present = self._tile_class_map.get(tile_id, [])
        for cls in present:
            if int(cls) not in self._visible_classes:
                continue
            cls_mask = (ann == cls).astype(np.uint8)
            if cls_mask.sum() == 0:
                continue
            # Split connected components into separate instances
            num_labels, labels = cv2.connectedComponents(cls_mask, connectivity=8)
            for label_id in range(1, num_labels):  # skip 0 (background)
                comp_mask = (labels == label_id)
                area = int(comp_mask.sum())
                if area < MIN_INSTANCE_AREA:
                    continue
                instances.append({
                    "category_id": int(cls),
                    "mask": torch.from_numpy(comp_mask),  # [H, W] bool
                })

        return {
            "image": image,
            "instances": instances,
            "image_id": index,
            "image_size": (H, W),
            "tile_id": tile_id,
            "classes": set(self._tile_class_map.get(tile_id, [])),
            "source_image": self._source_images.get(tile_id, tile_id),
        }

    # ── 查询接口 (SupportEpisodeQuery 协议) | Query interface ──

    def visible_classes(self) -> list[int]:
        """当前模式下的可见类别 ID 列表."""
        return sorted(self._class_tiles.keys())

    def class_to_tiles(self, class_id: int) -> list[int]:
        """某类的 tile 索引列表 | tile indices for a class."""
        return list(self._class_tiles.get(class_id, []))

    def source_image_id(self, idx: int) -> int:
        """tile 的来源图像 ID (用于场景不相交约束) | source image ID (for scene-disjoint)."""
        tile_id = self._tiles[idx]
        src = self._source_images.get(tile_id, tile_id)
        # Use hash for a stable integer ID
        return hash(src) & 0x7FFFFFFF

    # ── 辅助 | Helpers ──

    @property
    def tile_ids(self) -> list[str]:
        return list(self._tiles)

    def class_stats(self) -> dict[int, int]:
        """{class_id: tile_count} 统计."""
        return {c: len(t) for c, t in self._class_tiles.items()}


# ═══════════════════════════════════════════════════════════════════
# Episode Sampler (scene-disjoint for iSAID-5i)
# ═══════════════════════════════════════════════════════════════════

class ISAID5iEpisodeSampler:
    """iSAID-5i 场景不相交 episode 采样器 | Scene-disjoint episode sampler for iSAID-5i.

    约束: support 和 query 来自不同源图像 (避免数据泄漏)。
    Constraint: support and query from different source images (no leakage).

    :param dataset: ISAID5iDataset 实例.
    :param k_shot: 每个 episode 的 support tile 数.
    :param seed: 随机种子.
    :param min_tiles: 类别最少 tile 数 (低于则排除).
    """

    def __init__(
        self,
        dataset: ISAID5iDataset,
        k_shot: int = 5,
        seed: int = 42,
        min_tiles: int = 10,
    ) -> None:
        self.dataset = dataset
        self.k_shot = k_shot
        self.min_tiles = min_tiles
        self._rng = random.Random(seed)

        # Precompute: class → {source → [tile_idx]}
        self._class_scenes: dict[int, dict[str, list[int]]] = {}
        for cls in dataset.visible_classes():
            tiles = dataset.class_to_tiles(cls)
            if len(tiles) < min_tiles:
                continue
            scenes: dict[str, list[int]] = defaultdict(list)
            for idx in tiles:
                tile_id = dataset.tile_ids[idx]
                src = dataset._source_images.get(tile_id, tile_id)
                scenes[src].append(idx)
            if len(scenes) < 2:
                continue  # need ≥2 scenes for scene-disjoint
            self._class_scenes[cls] = dict(scenes)

        if not self._class_scenes:
            raise ValueError(
                f"No eligible class after filtering (min_tiles={min_tiles}, need ≥2 scenes)."
            )
        self._classes = sorted(self._class_scenes)

    def eligible_classes(self) -> list[int]:
        return list(self._classes)

    def sample(self) -> dict:
        """采样一个 episode | Sample one episode.

        :return: {"class_id", "support_indices"[≤K], "query_index"}.
        """
        cls = self._rng.choice(self._classes)
        scenes = self._class_scenes[cls]

        # Pick query scene + tile
        query_scene = self._rng.choice(list(scenes))
        query_index = self._rng.choice(scenes[query_scene])

        # Support from other scenes (scene-disjoint)
        support_pool = [
            idx for sid, idxs in scenes.items() if sid != query_scene for idx in idxs
        ]
        k = min(self.k_shot, len(support_pool))
        support_indices = self._rng.sample(support_pool, k)

        return {
            "class_id": cls,
            "support_indices": support_indices,
            "query_index": query_index,
        }
