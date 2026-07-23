"""
NEU_Seg 评估 (FastSAM-aligned) | Evaluation.
=============================================

用法 | Usage::
    python tools/eval_neuseg.py --checkpoint runs/neuseg_p3p4_k3_s42/best_model.pt
    python tools/eval_neuseg.py --checkpoint <ckpt> --save-vis
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import cv2, numpy as np, torch, torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from adasam.backbone import MultiScaleMobileSAMBackbone
from adasam.datasets import NEUSegDataset
from adasam.utils import set_seed
from tools.train_neuseg import PureDecoderP3P4, pad_to_32

CLASS_NAMES = NEUSegDataset.CLASS_NAMES
CLASS_COLORS = {0: (128,128,128), 1: (255,0,0), 2: (0,255,0), 3: (0,0,255)}


@torch.no_grad()
def eval_ckpt(backbone, decoder, dataset, device, num_classes=4):
    decoder.eval()
    inter = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    correct = 0; total = 0; per_sample = []

    for idx in range(len(dataset)):
        s = dataset[idx]
        img = s["image"]; gt = s["masks"].squeeze(0).long().to(device)
        H, W = gt.shape

        img_p, _, _ = pad_to_32(img.unsqueeze(0))
        feats = backbone(img_p.to(device))
        logits = decoder(feats["stage1"], feats["stage3"])
        pred = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        pred_c = pred.argmax(1).squeeze(0)

        correct += (pred_c == gt).sum().item(); total += gt.numel()
        sample_iou = 0.0; n_present = 0
        for c in range(num_classes):
            pc = (pred_c == c); gc = (gt == c)
            i = (pc & gc).sum(); u = (pc | gc).sum()
            inter[c] += i; union[c] += u
            if u > 0: sample_iou += (i/u).item(); n_present += 1
        per_sample.append({"image_id": s["image_id"],
                          "mIoU": round(sample_iou/max(n_present,1), 6),
                          "pixel_acc": round((pred_c==gt).float().mean().item(), 6)})

    ious = {}; valid = []
    for c in range(num_classes):
        i = inter[c].item(); u = union[c].item()
        iou = i/u if u > 0 else float("nan")
        ious[CLASS_NAMES[c]] = round(iou, 6)
        if iou == iou: valid.append(iou)
    sm = [s["mIoU"] for s in per_sample]
    return {"mIoU": round(float(np.mean(valid)),6) if valid else 0.0,
            "pixel_accuracy": round(correct/total,6) if total>0 else 0.0,
            "per_class_IoU": ious, "n_samples": len(per_sample),
            "sample_mIoU_mean": round(float(np.mean(sm)),6) if sm else 0.0,
            "sample_mIoU_median": round(float(np.median(sm)),6) if sm else 0.0,
            "per_sample": per_sample}


@torch.no_grad()
def save_vis(backbone, decoder, dataset, device, out_dir, num_classes=4, n=10):
    vis_dir = out_dir / "visualizations"; vis_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(min(n, len(dataset))):
        s = dataset[idx]; img = s["image"]
        gt = s["masks"].squeeze(0).long(); H, W = gt.shape
        img_p, _, _ = pad_to_32(img.unsqueeze(0))
        feats = backbone(img_p.to(device))
        logits = decoder(feats["stage1"], feats["stage3"])
        pred = F.interpolate(logits, size=(H,W), mode='bilinear', align_corners=False)
        pred_c = pred.argmax(1).squeeze(0).cpu()
        img_np = (img.permute(1,2,0).numpy()*255).astype(np.uint8)
        gt_c = gt.cpu().numpy()
        gt_col = np.zeros((H,W,3), np.uint8); pred_col = np.zeros((H,W,3), np.uint8)
        diff = np.zeros((H,W,3), np.uint8)
        for c, color in CLASS_COLORS.items():
            gt_col[gt_c==c] = color; pred_col[pred_c.numpy()==c] = color
        correct_mask = (gt_c == pred_c.numpy())
        diff[correct_mask] = (128,128,128); diff[~correct_mask] = (0,0,255)
        combined = np.hstack([img_np, gt_col, pred_col, diff])
        cv2.imwrite(str(vis_dir/f"{s['image_id']}.png"),
                    cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"  Visualizations saved to: {vis_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="data/NEU_Seg")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-vis", action="store_true")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed); device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists(): print(f"[ERROR] Not found: {ckpt_path}"); sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}); nc = ckpt.get("num_classes", 4)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent/"eval_neuseg"
    out_dir.mkdir(parents=True, exist_ok=True)

    mtype = cfg.get("backbone",{}).get("model_type","vit_t")
    bb_p = Path(cfg.get("backbone",{}).get("checkpoint","weights/mobile_sam.pt"))
    bb_p = bb_p if bb_p.is_absolute() else _REPO_ROOT/bb_p
    img_sz = int(cfg.get("backbone",{}).get("img_size", 224))
    backbone = MultiScaleMobileSAMBackbone.build(str(bb_p), mtype, device, img_size=img_sz)
    backbone.eval()

    d_cfg = cfg.get("decoder", {})
    decoder = PureDecoderP3P4(
        p3_ch=int(d_cfg.get("p3_channels",128)), p4_ch=int(d_cfg.get("p4_channels",256)),
        out_ch=nc, mid_ch=int(d_cfg.get("mid_channels",128))).to(device)
    decoder.load_state_dict(ckpt["decoder_state_dict"], strict=True)

    val_ds = NEUSegDataset(root=args.data_root, split="test")
    print(f"  Test: {len(val_ds)} samples")

    r = eval_ckpt(backbone, decoder, val_ds, device, num_classes=nc)
    print(f"  mIoU={r['mIoU']:.4f}  PA={r['pixel_accuracy']:.4f}  "
          f"mean={r['sample_mIoU_mean']:.4f}  median={r['sample_mIoU_median']:.4f}")
    for cn, iou_c in r["per_class_IoU"].items(): print(f"    {cn:>12s}: {iou_c:.4f}")

    with open(out_dir/"eval_results.json","w",encoding="utf-8") as f:
        json.dump({"checkpoint": str(ckpt_path), "num_classes": nc,
                   "results": r}, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_dir/'eval_results.json'}")

    if args.save_vis: save_vis(backbone, decoder, val_ds, device, out_dir, num_classes=nc)
    print("[Done]")


if __name__ == "__main__": main()
