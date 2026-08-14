"""Generate a screenshot-ready advisor update for LoveDA and Vaihingen."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "advisor_update_loveda_vaihingen.png"
FONT = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = FontProperties(fname=r"C:\Windows\Fonts\msyhbd.ttc")


def text(ax, x, y, value, size=18, color="#17202A", bold=False, **kwargs):
    ax.text(
        x, y, value, fontsize=size, color=color,
        fontproperties=FONT_BOLD if bold else FONT,
        transform=ax.transAxes, **kwargs,
    )


def panel(ax, x, y, w, h, color="#FFFFFF", edge="#DCE2E8"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.008",
        linewidth=1.2, edgecolor=edge, facecolor=color, transform=ax.transAxes,
    ))


def metric_table(ax, x, y, w, headers, rows, widths, row_h=0.036):
    colors = ["#F1F4F7", "#FFFFFF"]
    xpos = x
    for label, width in zip(headers, widths):
        ax.add_patch(plt.Rectangle((xpos, y), w * width, row_h, transform=ax.transAxes,
                                   facecolor="#263746", edgecolor="#263746"))
        text(ax, xpos + w * width / 2, y + row_h / 2, label, 13, "white", True,
             ha="center", va="center")
        xpos += w * width
    for row_index, row in enumerate(rows):
        yy = y - row_h * (row_index + 1)
        xpos = x
        for column, (value, width) in enumerate(zip(row, widths)):
            face = "#EAF6F0" if str(value).startswith("+") else colors[row_index % 2]
            ax.add_patch(plt.Rectangle((xpos, yy), w * width, row_h, transform=ax.transAxes,
                                       facecolor=face, edgecolor="#DCE2E8", linewidth=0.8))
            text(ax, xpos + w * width / 2, yy + row_h / 2, str(value), 13,
                 "#12734A" if str(value).startswith("+") else "#263746",
                 str(value).startswith("+"), ha="center", va="center")
            xpos += w * width


def main() -> None:
    fig = plt.figure(figsize=(16, 9), dpi=150, facecolor="#F5F7F9")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    text(ax, 0.045, 0.935, "阶段性结果：语义条件预算化表示分配", 28, "#15232E", True)
    text(ax, 0.045, 0.895, "结论：小目标表示选择已有稳定信号，但跨数据集泛化与实际加速仍未解决", 17, "#50616F")

    panel(ax, 0.04, 0.53, 0.445, 0.33)
    text(ax, 0.06, 0.815, "LoveDA", 22, "#A0442B", True)
    text(ax, 0.155, 0.817, "10%标注 · 冻结式适配 · seed 42", 13, "#6C757D")
    text(ax, 0.06, 0.775, "旧 Adaptive 未学到可靠分配策略", 16, "#A0442B", True)
    metric_table(
        ax, 0.06, 0.695, 0.405,
        ["25%策略", "mIoU", "Small IoU", "相对Adaptive"],
        [
            ["Adaptive", "37.21%", "10.98%", "基准"],
            ["Random", "40.37%", "13.65%", "+3.16"],
            ["Magnitude", "42.09%", "14.25%", "+4.88"],
        ],
        [0.27, 0.22, 0.26, 0.25], row_h=0.038,
    )
    text(ax, 0.06, 0.558, "补充证据", 14, "#263746", True)
    text(ax, 0.145, 0.558, "P4+Embedding 39.59% ≈ 全层级 39.48%；75%预算接近100%性能", 12.5, "#50616F")

    panel(ax, 0.515, 0.53, 0.445, 0.33)
    text(ax, 0.535, 0.815, "Vaihingen", 22, "#146B4A", True)
    text(ax, 0.665, 0.817, "100%训练区域 · LoRA · 3 seeds", 13, "#6C757D")
    text(ax, 0.535, 0.775, "Adaptive 对小目标的提升稳定复现", 16, "#146B4A", True)
    metric_table(
        ax, 0.535, 0.695, 0.405,
        ["25%策略", "mIoU-5", "Small IoU", "Car IoU"],
        [
            ["Random", "68.35%", "22.82%", "47.78%"],
            ["Magnitude", "68.81%", "24.35%", "50.07%"],
            ["Adaptive", "69.43%", "25.30%", "54.42%"],
        ],
        [0.27, 0.23, 0.25, 0.25], row_h=0.038,
    )
    text(ax, 0.535, 0.558, "三种子一致", 14, "#146B4A", True)
    text(ax, 0.635, 0.558, "Adaptive 每个 seed 均超过 Random 与 Magnitude", 12.5, "#50616F")

    panel(ax, 0.04, 0.265, 0.92, 0.215)
    text(ax, 0.06, 0.435, "效率核查", 20, "#263746", True)
    text(ax, 0.06, 0.395, "25%空间预算真实执行", 15, "#146B4A", True)
    text(ax, 0.235, 0.395, "P3: 1024/4096   ·   P4: ≈255/1024   ·   局部投影计算 -75%", 14, "#344955")
    text(ax, 0.06, 0.350, "但总计算收益很小", 15, "#A0442B", True)
    text(ax, 0.235, 0.350, "总FLOPs仅 -0.63%   ·   FPS未提升（Adaptive 29.50 vs Magnitude 31.46）", 14, "#344955")
    text(ax, 0.06, 0.303, "原因", 15, "#263746", True)
    text(ax, 0.125, 0.303, "路由发生在 MobileSAM 完成特征提取之后，只压缩后续侧向投影，未跳过主干计算。", 14, "#50616F")

    panel(ax, 0.04, 0.055, 0.44, 0.16, color="#EAF6F0", edge="#B9DACB")
    text(ax, 0.06, 0.175, "目前能够支持", 17, "#146B4A", True)
    text(ax, 0.06, 0.135, "• 不同层级对不同尺度目标的价值不同", 13.5, "#244C3E")
    text(ax, 0.06, 0.101, "• 固定预算下，语义条件选择改善小目标", 13.5, "#244C3E")
    text(ax, 0.06, 0.067, "• Vaihingen 三种子结果稳定", 13.5, "#244C3E")

    panel(ax, 0.505, 0.055, 0.455, 0.16, color="#FFF3ED", edge="#E7C6B8")
    text(ax, 0.525, 0.175, "尚未解决 / 请老师建议", 17, "#A0442B", True)
    text(ax, 0.525, 0.135, "× LoveDA低标注条件下 Adaptive 泛化失败", 13.5, "#6D3D31")
    text(ax, 0.525, 0.101, "× 当前实现没有形成实际速度优势", 13.5, "#6D3D31")
    text(ax, 0.525, 0.067, "选择：收敛为“表示选择”，还是将路由前移到TinyViT内部？", 13.5, "#6D3D31", True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
