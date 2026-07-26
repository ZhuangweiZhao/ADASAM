#!/usr/bin/env python
"""
MobileSAM 交互式点提示分割工具 | Interactive Point-Prompt Segmentation Tool
=============================================================================

使用 MobileSAM 对任意图像进行交互式点提示分割。
左键添加前景点 (绿色), 右键添加背景点 (红色), 实时显示分割结果。

Usage:
    python tools/interactive_point_prompt.py                          # GUI 选文件夹
    python tools/interactive_point_prompt.py --image path/to/img.jpg  # 单张图像
    python tools/interactive_point_prompt.py --dir path/to/folder     # 指定文件夹
    python tools/interactive_point_prompt.py --dir . --device cpu     # CPU 模式

Controls:
    左键点击    — 添加前景点 (foreground, label=1) 绿色 ×
    右键点击    — 添加背景点 (background, label=0) 红色 +
    c / C       — 清除所有点
    s / S       — 切换分割叠加显示 (on/off)
    1 / 2 / 3   — 手动切换 multi-mask 输出 (SAM 返回3个候选mask)
    n / →       — 下一张图像
    p / ←       — 上一张图像
    r           — 重新加载当前图像 (重置所有状态)
    h           — 打印帮助到终端
    q           — 退出
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("TkAgg")  # must be before importing pyplot

# ---- Inject vendored MobileSAM ----
_MOBILE_SAM_ROOT = Path(__file__).resolve().parents[1] / "thirdparty" / "MobileSAM"
if str(_MOBILE_SAM_ROOT) not in sys.path:
    sys.path.insert(0, str(_MOBILE_SAM_ROOT))

from mobile_sam import sam_model_registry, SamPredictor  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FG_COLOR = "#00ff00"       # green  — foreground point
BG_COLOR = "#ff0000"       # red    — background point
MASK_COLOR = np.array([255, 255, 0]) / 255.0   # yellow overlay
FG_MARKER = "x"
BG_MARKER = "+"
MARKER_SIZE = 140
ALPHA = 0.45

HELP_TEXT = """
Controls:
  Left Click     — 添加前景点 (绿色 ×)
  Right Click    — 添加背景点 (红色 +)
  c              — 清除所有点
  s              — 切换分割叠加
  1/2/3          — 切换 multi-mask
  n / Right      — 下一张
  p / Left       — 上一张
  r              — 重新加载
  h              — 帮助
  q              — 退出
"""

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def _collect_images(path: str | Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    root = Path(path)
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)


def _select_folder_gui() -> Optional[str]:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="选择图像文件夹 | Select Image Folder")
    root.destroy()
    return folder if folder else None


# ---------------------------------------------------------------------------
# Interactive Session
# ---------------------------------------------------------------------------

class InteractivePointPrompt:
    """MobileSAM 交互式点提示会话 | Interactive point-prompt session with MobileSAM."""

    def __init__(
        self,
        checkpoint: str | Path = "weights/mobile_sam.pt",
        device: str = "cuda",
        show_mask_on_start: bool = True,
    ):
        print(f"[MobileSAM] 加载模型 {checkpoint} ...")
        self.device = device if torch.cuda.is_available() else "cpu"
        if self.device != device:
            print(f"  ⚠ CUDA 不可用, 回退到 {self.device}")

        sam = sam_model_registry["vit_t"](checkpoint=str(checkpoint))
        sam.to(self.device)
        sam.eval()
        self.predictor = SamPredictor(sam)
        print("  ✓ 模型加载完成")

        # Image list state
        self.images: list[Path] = []
        self.idx: int = 0

        # Per-image state
        self.img_original: np.ndarray | None = None      # H×W×3  RGB  uint8
        self.points: list[tuple[float, float, int]] = []  # (x, y, label) in image pixels
        self.masks: np.ndarray | None = None               # C×H×W
        self.iou_scores: np.ndarray | None = None          # C
        self.show_mask: bool = show_mask_on_start
        self.mask_idx: int = 0

        # Matplotlib
        self.fig: plt.Figure | None = None
        self.ax: plt.Axes | None = None
        self.img_artist: plt.Axes | None = None
        self.fg_scatter: plt.PathCollection | None = None
        self.bg_scatter: plt.PathCollection | None = None
        self.mask_overlay: plt.Axes | None = None
        self.contour_lines: list = []
        self.title: plt.Text | None = None

    # ---- data loading ----------------------------------------------------

    def load_folder(self, folder: str | Path) -> int:
        self.images = _collect_images(folder)
        print(f"[Images] 找到 {len(self.images)} 张图像")
        if not self.images:
            print("  ⚠ 未找到图像文件")
        return len(self.images)

    def load_image(self, path: str | Path) -> int:
        path = Path(path)
        if path.is_file():
            self.images = [path]
            print(f"[Images] 单张: {path.name}")
            return 1
        elif path.is_dir():
            return self.load_folder(path)
        return 0

    # ---- per-image processing --------------------------------------------

    def set_current_image(self) -> None:
        if self.idx < 0 or self.idx >= len(self.images):
            return
        path = self.images[self.idx]
        print(f"[{self.idx+1}/{len(self.images)}] {path.name}")
        img_bgr = plt.imread(str(path))
        if img_bgr.ndim == 2:
            img_rgb = np.stack([img_bgr] * 3, axis=-1)
        else:
            img_rgb = img_bgr[..., :3]  # discard alpha if present

        # If uint8 [0,255], keep; if float [0,1], convert to uint8
        if img_rgb.dtype == np.float32 or img_rgb.dtype == np.float64:
            if img_rgb.max() <= 1.0:
                img_rgb = (img_rgb * 255).astype(np.uint8)
            else:
                img_rgb = img_rgb.astype(np.uint8)

        self.img_original = img_rgb
        self.img_h, self.img_w = img_rgb.shape[:2]
        self.predictor.set_image(img_rgb, image_format="RGB")
        self.points.clear()
        self.masks = None
        self.iou_scores = None
        self._rebuild_plot()

    def _predict(self) -> None:
        if not self.points:
            self.masks = None
            self.iou_scores = None
            return

        coords = np.array([[x, y] for x, y, _ in self.points], dtype=np.float32)
        labels = np.array([lbl for _, _, lbl in self.points], dtype=np.int32)

        masks, scores, _ = self.predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=True,
        )
        self.masks = masks
        self.iou_scores = scores
        if self.mask_idx >= masks.shape[0]:
            self.mask_idx = 0

    # ---- matplotlib drawing ----------------------------------------------

    def _rebuild_plot(self) -> None:
        """Update all plot elements in-place without recreating the figure."""
        if self.fig is None or self.ax is None:
            return

        ax = self.ax
        img = self.img_original.copy()
        H, W = img.shape[:2]

        # --- Run prediction ---
        self._predict()

        # --- Update base image ---
        if self.img_artist is None:
            self.img_artist = ax.imshow(img, extent=(0, W, H, 0), animated=False)
        else:
            # Blend mask overlay onto a copy of the original for the image artist
            display = img.astype(np.float64) / 255.0
            if self.show_mask and self.masks is not None and len(self.masks) > 0:
                mask = self.masks[self.mask_idx]
                overlay = np.zeros_like(display)
                overlay[mask] = MASK_COLOR
                display = display * (1 - ALPHA) + overlay * ALPHA
            self.img_artist.set_data(display)

        # --- Remove old contour lines ---
        for coll in self.contour_lines:
            try:
                coll.remove()
            except Exception:
                pass
        self.contour_lines.clear()

        # --- Draw mask contours ---
        if self.show_mask and self.masks is not None and len(self.masks) > 0:
            mask = self.masks[self.mask_idx]
            import matplotlib.path as mpath
            try:
                # Generate contours using matplotlib's contour
                h, w = mask.shape
                X, Y = np.meshgrid(np.linspace(0, W, w + 1)[:w], np.linspace(0, H, h + 1)[:h])
                cs = ax.contour(
                    X, Y, mask.astype(float),
                    levels=[0.5], colors="cyan", linewidths=1.5
                )
                self.contour_lines.append(cs)
            except Exception:
                # fallback: use connected components edge
                pass

        # --- Update point scatter ---
        fg_pts = [(x, y) for x, y, lbl in self.points if lbl == 1]
        bg_pts = [(x, y) for x, y, lbl in self.points if lbl == 0]

        # Clear old scatter artists
        if self.fg_scatter is not None:
            self.fg_scatter.remove()
        if self.bg_scatter is not None:
            self.bg_scatter.remove()

        if fg_pts:
            self.fg_scatter = ax.scatter(
                [p[0] for p in fg_pts], [p[1] for p in fg_pts],
                c=FG_COLOR, marker=FG_MARKER, s=MARKER_SIZE,
                linewidths=2.5, zorder=10, edgecolors="white"
            )
        else:
            self.fg_scatter = None

        if bg_pts:
            self.bg_scatter = ax.scatter(
                [p[0] for p in bg_pts], [p[1] for p in bg_pts],
                c=BG_COLOR, marker=BG_MARKER, s=MARKER_SIZE,
                linewidths=2.5, zorder=10, edgecolors="white"
            )
        else:
            self.bg_scatter = None

        # --- Title / info ---
        n_fg = sum(1 for _, _, lbl in self.points if lbl == 1)
        n_bg = sum(1 for _, _, lbl in self.points if lbl == 0)
        name = self.images[self.idx].name if self.images else ""
        short_name = name if len(name) <= 60 else name[:57] + "..."

        lines = [
            f"[{self.idx+1}/{len(self.images)}] {short_name}  ({W}×{H})",
            f"Points: {len(self.points)} (fg:{n_fg} bg:{n_bg})  |  "
            f"Mask: {'ON' if self.show_mask else 'OFF'} [{self.mask_idx+1}]",
        ]
        if self.iou_scores is not None and len(self.iou_scores) > 0:
            lines[1] += f"  |  IoU est: {self.iou_scores[self.mask_idx]:.4f}"

        title_str = "\n".join(lines)
        if self.title is None:
            self.title = ax.set_title(title_str, fontsize=10, fontfamily="monospace",
                                       loc="left", pad=2)
        else:
            self.title.set_text(title_str)

        self.fig.canvas.draw_idle()

    # ---- event handlers --------------------------------------------------

    def _on_click(self, event) -> None:
        """Handle mouse clicks on the axes."""
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x, y = event.xdata, event.ydata
        if event.button == 1:       # left → foreground
            self.points.append((x, y, 1))
        elif event.button == 3:     # right → background
            self.points.append((x, y, 0))
        else:
            return

        self._rebuild_plot()

    def _on_key(self, event) -> None:
        """Handle keypress events."""
        key = event.key

        if key in ("q",):
            plt.close(self.fig)
            return

        elif key == "h":
            print(HELP_TEXT)

        elif key in ("c", "C"):
            self.points.clear()
            self.masks = None
            self.iou_scores = None
            self._rebuild_plot()
            print("  → 清除所有点 | Cleared all points")

        elif key in ("s", "S"):
            self.show_mask = not self.show_mask
            self._rebuild_plot()
            print(f"  → 分割叠加: {'ON' if self.show_mask else 'OFF'}")

        elif key in ("1", "2", "3"):
            idx = int(key) - 1
            if self.masks is not None and idx < self.masks.shape[0]:
                self.mask_idx = idx
                self._rebuild_plot()
                print(f"  → Multi-mask #{idx+1}  IoU est: {self.iou_scores[idx]:.4f}")

        elif key in ("n", "right"):
            self.idx = (self.idx + 1) % len(self.images)
            self.set_current_image()

        elif key in ("p", "left"):
            self.idx = (self.idx - 1) % len(self.images)
            self.set_current_image()

        elif key == "r":
            print("  → 重新加载当前图像")
            self.set_current_image()

    # ---- main loop -------------------------------------------------------

    def run(self) -> None:
        if not self.images:
            print("[ERROR] 没有图像可显示, 请先用 --dir 或 --image 加载")
            return

        self.idx = 0

        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("MobileSAM Point Prompt — 交互分割")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Connect events
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Load first image
        self.set_current_image()

        plt.tight_layout(pad=0.5)

        print(HELP_TEXT)
        print(f"共 {len(self.images)} 张图像, 从第 1 张开始.")
        print("左键=前景点  右键=背景点  h=帮助  q=退出")
        plt.show()
        print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MobileSAM Interactive Point-Prompt Segmentation"
    )
    parser.add_argument("--image", type=str, default=None, help="单张图像路径")
    parser.add_argument("--dir", type=str, default=None, help="图像文件夹路径")
    parser.add_argument("--checkpoint", type=str, default="weights/mobile_sam.pt",
                        help="MobileSAM 权重路径")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="推理设备 (default: cuda)")
    parser.add_argument("--no-mask", action="store_true", help="启动时不显示分割叠加")
    args = parser.parse_args()

    # Resolve image source
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"[ERROR] 图像不存在: {image_path}")
            sys.exit(1)
        source = str(image_path)
        source_type = "image"
    elif args.dir:
        source = args.dir
        source_type = "dir"
    else:
        folder = _select_folder_gui()
        if not folder:
            print("[EXIT] 未选择文件夹 | No folder selected.")
            sys.exit(0)
        source = folder
        source_type = "dir"

    session = InteractivePointPrompt(
        checkpoint=args.checkpoint,
        device=args.device,
        show_mask_on_start=not args.no_mask,
    )

    if source_type == "image":
        n = session.load_image(source)
    else:
        n = session.load_folder(source)

    if n == 0:
        print("[ERROR] 未找到可显示的图像")
        sys.exit(1)

    session.run()


if __name__ == "__main__":
    main()
