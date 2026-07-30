from __future__ import annotations
import argparse
import random
from pathlib import Path
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

# ============ 路径配置（适配你的工程） ============
REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adasam.adapters import CATAdapter
from adasam.backbone import MobileSAMBackbone, build_mobile_sam
from adasam.datasets import NEUSegDataset
from adasam.model import AdaSAMModel, AdaSAMModelConfig
from adasam.utils import set_seed
from adasam.utils.transforms import preprocess_image, resize_mask

FOREGROUND_CLASSES = (1, 2, 3)
COLOR_MAP = {
    0: np.array([0, 0, 0]),       # background black
    1: np.array([255, 60, 60]),  # class1 red
    2: np.array([60, 220, 60]),  # class2 green
    3: np.array([60, 120, 255]), # class3 blue
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_checkpoint(ckpt_path: Path, device: torch.device):
    """自动识别 stage1 / stage2 权重"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    stage_tag = ckpt.get("stage")
    if stage_tag not in ["neuseg_stage1", "neuseg_stage2"]:
        raise RuntimeError(f"Unknown checkpoint stage tag: {stage_tag}")
    return ckpt, stage_tag


def build_model_from_ckpt(ckpt, backbone_ckpt_path: Path, device):
    # 构建MobileSAM主干
    sam = build_mobile_sam(str(backbone_ckpt_path), model_type="vit_t", device=device)
    backbone = MobileSAMBackbone(sam.image_encoder, sam.image_encoder.img_size).to(device)

    # 加载CATAdapter
    cfg_adapter = ckpt.get("config", {}).get("adapter", {})
    adapter = CATAdapter(
        dim=256,
        bottleneck=int(cfg_adapter.get("bottleneck", 64))
    ).to(device)
    adapter.load_state_dict(ckpt["adapter"])
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad = False

    # 构建AdaSAM
    model_cfg = AdaSAMModelConfig.from_dict(ckpt["config"])
    model = AdaSAMModel(sam, model_cfg).to(device)

    # stage2含有完整model权重，stage1只有adapter
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    model.eval()
    return sam, backbone, adapter, model


@torch.no_grad()
def extract_feature(image, backbone, adapter, device):
    proc_img, meta = preprocess_image(image)
    img_tensor = proc_img.unsqueeze(0).to(device)
    feat = backbone(img_tensor)["image_embedding"]
    feat = adapter(feat)
    return feat, meta


def mask_to_rgb(mask_2d: np.ndarray) -> np.ndarray:
    """类别掩码转彩色图"""
    h, w = mask_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in COLOR_MAP.items():
        rgb[mask_2d == cls_id] = color
    return rgb


def overlay_mask(image_rgb: np.ndarray, mask_rgb: np.ndarray, alpha=0.5):
    """掩码叠加原图"""
    return cv2.addWeighted(image_rgb, 1.0, mask_rgb, alpha, 0)


@torch.no_grad()
def visualize_one_episode(
    ds: NEUSegDataset,
    class_id: int,
    support_indices: list[int],
    query_idx: int,
    backbone,
    adapter,
    model,
    device,
    save_path: Path,
    threshold=0.5
):
    # === 1. 加载support & query样本
    support_feats = []
    support_masks = []
    for sid in support_indices:
        s_sample = ds[sid]
        s_feat, _ = extract_feature(s_sample["image"], backbone, adapter, device)
        support_feats.append(s_feat[0])
        mask_bin = resize_mask((s_sample["masks"].squeeze(0) == class_id), (64, 64))
        support_masks.append(mask_bin)

    support_feats = torch.stack(support_feats).to(device)
    support_masks = torch.stack(support_masks).to(device)

    q_sample = ds[query_idx]
    query_feat, meta = extract_feature(q_sample["image"], backbone, adapter, device)
    img_origin = q_sample["image"].numpy().transpose(1, 2, 0)
    img_origin = (img_origin * 255).astype(np.uint8)
    gt_mask = q_sample["masks"].squeeze(0).numpy().astype(np.int32)

    # === 2. 前向推理【修复维度】
    _, low_res, _ = model.forward_train(query_feat, support_feats, support_masks)
    logits = model.sam_decoder.upscale_logits(low_res, meta.input_size, q_sample["image_size"])

    # 打印维度，方便调试
    print(f"low_res shape: {low_res.shape}")
    print(f"logits shape: {logits.shape}")

    # 稳妥方式：先detach转到cpu，再去除batch、通道维，保证输出2D数组
    prob_tensor = logits.sigmoid().detach().cpu()
    prob_np = prob_tensor.numpy()
    # 去除batch维度、单通道维度
    prob = np.squeeze(prob_np)

    print(f"prob final shape: {prob.shape}")

    # === 3. 单类别预测掩码（当前episode只预测目标class）
    pred_bin = (prob > threshold).astype(np.int32)
    pred_mask = np.zeros_like(gt_mask)
    pred_mask[pred_bin > 0] = class_id

    # === 4. 绘图
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    axes[0].set_title("Query Image")
    axes[0].imshow(img_origin)
    axes[0].axis("off")

    axes[1].set_title(f"GT Mask | {ds.CLASS_NAMES[class_id]}")
    gt_rgb = mask_to_rgb(gt_mask)
    axes[1].imshow(overlay_mask(img_origin, gt_rgb))
    axes[1].axis("off")

    axes[2].set_title(f"Pred Mask | {ds.CLASS_NAMES[class_id]}")
    pred_rgb = mask_to_rgb(pred_mask)
    axes[2].imshow(overlay_mask(img_origin, pred_rgb))
    axes[2].axis("off")

    # support样本展示（只展示第一个support图）
    s0_sample = ds[support_indices[0]]
    s_img = s0_sample["image"].numpy().transpose(1,2,0)
    s_img = (s_img * 255).astype(np.uint8)
    s_mask_bin = (s0_sample["masks"].squeeze(0) == class_id).numpy().astype(np.int32)
    s_rgb = mask_to_rgb(s_mask_bin)
    axes[3].set_title("Support Image & Mask")
    axes[3].imshow(overlay_mask(s_img, s_rgb))
    axes[3].axis("off")

    axes[4].set_title("Pred Probability Map")
    axes[4].imshow(prob, cmap="jet")
    axes[4].axis("off")

    axes[5].set_title("Pred vs GT Overlay")
    compare = np.hstack([gt_rgb, pred_rgb])
    axes[5].imshow(compare)
    axes[5].axis("off")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization -> {save_path}")


def main():
    parser = argparse.ArgumentParser("NEUSeg Stage1/Stage2 prediction visualization")
    parser.add_argument("--ckpt", required=True, help="stage1 or stage2 checkpoint path")
    parser.add_argument("--backbone-ckpt", required=True, help="MobileSAM vit-t weight")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--support-shot", type=int, default=1)
    parser.add_argument("--num-vis", type=int, default=5, help="visualize how many episodes")
    parser.add_argument("--output", default="./vis_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = resolve_path(args.output)
    out_dir.mkdir(exist_ok=True, parents=True)

    # 数据集
    ds = NEUSegDataset(resolve_path(args.data_root), split=args.split)
    print(f"Dataset loaded, split={args.split}, total samples: {len(ds)}")

    # 加载权重
    ckpt, stage_tag = load_checkpoint(resolve_path(args.ckpt), device)
    print(f"Detected checkpoint: {stage_tag}")
    sam, backbone, adapter, model = build_model_from_ckpt(
        ckpt, resolve_path(args.backbone_ckpt), device
    )

    # 预收集：每个类别对应的样本索引
    class_sample_map = {cid: [] for cid in FOREGROUND_CLASSES}
    for idx in range(len(ds)):
        mask = ds[idx]["masks"]
        unique_cls = torch.unique(mask)
        for cid in FOREGROUND_CLASSES:
            if (unique_cls == cid).any():
                class_sample_map[cid].append(idx)

    # 循环可视化N个episode
    for vis_id in range(args.num_vis):
        cls_id = random.choice(FOREGROUND_CLASSES)
        all_cls_samples = class_sample_map[cls_id]
        if len(all_cls_samples) < 2:
            query_idx = all_cls_samples[0]
            support_indices = [query_idx]
        else:
            query_idx = random.choice(all_cls_samples)
            pool = [x for x in all_cls_samples if x != query_idx]
            support_indices = random.sample(pool, min(args.support_shot, len(pool)))

        save_name = f"vis_{vis_id:03d}_cls{cls_id}_q{query_idx}.png"
        save_file = out_dir / save_name
        visualize_one_episode(
            ds=ds,
            class_id=cls_id,
            support_indices=support_indices,
            query_idx=query_idx,
            backbone=backbone,
            adapter=adapter,
            model=model,
            device=device,
            save_path=save_file,
            threshold=args.threshold
        )

    print(f"\nVisualization finished! Output folder: {out_dir}")


if __name__ == "__main__":
    main()
