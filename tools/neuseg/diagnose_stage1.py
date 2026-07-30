"""
Diagnosis tools for NEU_Seg Stage1 adapter (domain adaptation) quality.

Modes supported:
  - correlation : correlation heatmaps between support prototype and query features
                   (Peak-to-GT distance, Correlation IoU @ thresholds, Localization Recall@K)
  - cosine      : prototype cosine distributions (positive vs negative pairs)
  - retrieval   : prototype retrieval metrics (Recall@k)
  - tsne        : t-SNE / UMAP visualization of pixel / patch embeddings (before/after adapter)
  - all         : run all enabled diagnostics

Usage examples:
  python tools/neuseg/diagnose_stage1.py --mode correlation \
      --stage1-ckpt runs/neuseg_stage1_k5_seed42/best_adapter.pt \
      --backbone-ckpt weights/mobile_sam.pt --data-root ./data/NEU_Seg \
      --n-episodes 100 --outdir runs/diag_corr --device cuda

  python tools/neuseg/diagnose_stage1.py --mode tsne \
      --stage1-ckpt runs/neuseg_stage1_k5_seed42/best_adapter.pt \
      --backbone-ckpt weights/mobile_sam.pt --data-root ./data/NEU_Seg \
      --n-samples 400 --outdir runs/diag_tsne --device cuda

Outputs: PNG plots and JSON metrics saved under the --outdir.

"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# sklearn components (t-SNE, PCA)
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except Exception:
    PCA = None
    TSNE = None

# Repo imports (reuse existing code)
_REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adasam.backbone import build_mobile_sam, MultiScaleMobileSAMBackbone, MobileSAMBackbone
from adasam.adapters import CATAdapter
from adasam.datasets import NEUSegDataset
from adasam.utils.transforms import preprocess_image, resize_mask


# ----------------- Utilities -----------------

def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else _REPO_ROOT / p


def load_stage1_adapter(path: Path, device: torch.device) -> Tuple[CATAdapter, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "adapter" not in ckpt:
        raise RuntimeError(f"Stage1 checkpoint {path} has no 'adapter' key")
    adapter_cfg = ckpt.get("config", {}).get("adapter", {})
    bottleneck = int(adapter_cfg.get("bottleneck", 64))
    adapter = CATAdapter(dim=256, bottleneck=bottleneck).to(device)
    adapter.load_state_dict(ckpt["adapter"])  # may raise if mismatch
    adapter.eval()
    manifest = ckpt.get("kshot_manifest")
    return adapter, manifest or {}


def build_backbone(backbone_ckpt: Path, device: torch.device, img_size: int | None = None):
    # build MultiScale or MobileSAM backbone per availability
    if img_size is None:
        sam = build_mobile_sam(str(backbone_ckpt), model_type="vit_t", device=device)
        backbone = MobileSAMBackbone(sam.image_encoder, sam.image_encoder.img_size).to(device)
    else:
        backbone = MultiScaleMobileSAMBackbone.build(str(backbone_ckpt), model_type="vit_t", device=device, img_size=img_size).to(device)
    backbone.eval()
    return backbone


@torch.no_grad()
def extract_adapter_feature(image: torch.Tensor, backbone, adapter, device) -> Tuple[torch.Tensor, object]:
    """Return adapter-applied feature [1, C, gh, gw] and preprocess meta."""
    proc, meta = preprocess_image(image)
    proc_b = proc.unsqueeze(0).to(device)
    feats = backbone(proc_b)
    # MobileSAMBackbone returns {"image_embedding": [B,256,64,64]}
    if isinstance(feats, dict) and "image_embedding" in feats:
        emb = feats["image_embedding"]
    else:
        # Multi-scale returns dict with stage3 etc — prefer stage3 or image_embedding
        emb = feats.get("stage3") if "stage3" in feats else list(feats.values())[-1]
    adapted = adapter(emb)
    return adapted.detach(), meta


def prototype_from_supports(support_features: torch.Tensor, support_masks: torch.Tensor) -> torch.Tensor:
    """Compute masked-mean prototype from K supports.
    support_features: [K, C, gh, gw]
    support_masks: [K, gh, gw] (0/1)
    returns: prototype [C]
    """
    K = support_features.shape[0]
    masked = support_features * support_masks.unsqueeze(1)
    fg_sum = masked.sum(dim=(0, 2, 3))  # [C]
    fg_count = support_masks.sum() + 1e-8
    proto = fg_sum / fg_count
    return proto


# ----------------- Mode: correlation -----------------

@torch.no_grad()
def run_correlation(backbone, adapter, stage1_manifest, data_root: Path, device: torch.device, outdir: Path, n_episodes: int = 100, support_modes: List[str] = ["manifest", "random"], seed: int = 42):
    outdir.mkdir(parents=True, exist_ok=True)
    ds_train = NEUSegDataset(data_root, split="train")
    ds_val = NEUSegDataset(data_root, split="test")
    # mapping name->idx
    name2idx_train = {n: i for i, n in enumerate(ds_train.sample_names)}

    manifest_ids = []
    if stage1_manifest and "sample_ids" in stage1_manifest:
        for n in stage1_manifest["sample_ids"]:
            if n in name2idx_train:
                manifest_ids.append(name2idx_train[n])
    rng = random.Random(seed)

    # pre-cache features for train selected (manifest) and optionally random pool
    cache = {}
    def get_feature_cached(ds, idx):
        if (ds, idx) in cache:
            return cache[(ds, idx)]
        feat, meta = extract_adapter_feature(ds[idx]["image"], backbone, adapter, device)
        cache[(ds, idx)] = (feat.cpu(), meta)
        return cache[(ds, idx)]

    results = {mode: [] for mode in support_modes}
    thresholds = [0.2, 0.3, 0.4, 0.5]

    for ep in range(n_episodes):
        q_idx = rng.randrange(len(ds_val))
        q_sample = ds_val[q_idx]
        q_feat, q_meta = extract_adapter_feature(q_sample["image"], backbone, adapter, device)
        gh, gw = q_feat.shape[2], q_feat.shape[3]
        q_mask_grid = resize_mask(q_sample["masks"].squeeze(0).float(), (gh, gw))
        q_mask_np = q_mask_grid.numpy()
        # if query has no foreground, skip
        if q_mask_np.sum() == 0:
            continue
        for mode in support_modes:
            # select supports
            if mode == "manifest" and manifest_ids:
                # pick supports from manifest that contain the target class when possible
                # find class id present in query
                present = np.unique(q_sample["masks"].numpy())
                present = [int(x) for x in present if int(x) != 0]
                if not present:
                    continue
                cls = present[0]
                # find manifest indices that contain this class
                cand = []
                for idx in manifest_ids:
                    labels = np.unique(ds_train[idx]["masks"].numpy())
                    if cls in labels:
                        cand.append(idx)
                if not cand:
                    # fallback to random
                    cand = list(range(len(ds_train)))
                # sample up to 5 supports
                support_indices = rng.sample(cand, min(5, len(cand)))
            else:
                # random supports drawn from train
                support_indices = rng.sample(range(len(ds_train)), min(5, len(ds_train)))

            # build support tensors
            s_feats = []
            s_masks = []
            for si in support_indices:
                s_feat, s_meta = get_feature_cached(ds_train, si)
                # resize mask to gh,gw
                s_mask = resize_mask(ds_train[si]["masks"].squeeze(0).float(), (gh, gw))
                s_feats.append(s_feat[0])
                s_masks.append(s_mask)
            s_feats = torch.stack(s_feats).to(device)
            s_masks = torch.stack(s_masks).to(device)
            proto = prototype_from_supports(s_feats, s_masks)  # [C]
            # normalize
            p_norm = F.normalize(proto.view(1, -1), dim=1)  # [1, C]
            q_flat = q_feat.view(1, q_feat.shape[1], -1)  # [1, C, N]
            q_norm = F.normalize(q_flat, dim=1)
            sim = torch.einsum('bcn,bc->bn', q_norm, p_norm)  # [1, N]
            sim_map = sim.view(gh, gw).cpu().numpy()

            # metrics: peak position -> distance to GT centroid (in pixels)
            peak_idx = np.unravel_index(int(sim_map.argmax()), sim_map.shape)
            # GT centroid in grid coords
            ys, xs = np.nonzero(q_mask_np)
            if len(ys) == 0:
                centroid = (gh//2, gw//2)
            else:
                centroid = (int(ys.mean()), int(xs.mean()))
            # grid -> pixel scale
            orig_h, orig_w = q_meta.original_size
            pix_h = orig_h / gh
            pix_w = orig_w / gw
            dy = (peak_idx[0] - centroid[0]) * pix_h
            dx = (peak_idx[1] - centroid[1]) * pix_w
            peak_to_centroid = math.sqrt(dx*dx + dy*dy)

            # IoU @ thresholds (on grid)
            ious = {}
            for t in thresholds:
                bin_map = (sim_map > t).astype(np.uint8)
                inter = int(((bin_map==1) & (q_mask_np==1)).sum())
                union = int(((bin_map==1) | (q_mask_np==1)).sum())
                iou = float(inter / union) if union>0 else 0.0
                ious[f"iou_t{t}"] = iou

            # Localization Recall@K (top K peaks)
            flat_idx = np.argsort(sim_map.ravel())[::-1]
            recalls = {}
            for K in [1,5,10,20]:
                topk = flat_idx[:K]
                topk_coords = np.vstack([topk // gw, topk % gw]).T
                hit = any(q_mask_np[c[0], c[1]]==1 for c in topk_coords)
                recalls[f"recall@{K}"] = 1.0 if hit else 0.0

            # Save heatmap visualization
            fig, axes = plt.subplots(1,3,figsize=(12,4))
            img_np = (q_sample["image"].permute(1,2,0).numpy()*255).astype(np.uint8)
            axes[0].imshow(img_np); axes[0].set_title("Query Image"); axes[0].axis('off')
            axes[1].imshow(q_mask_np, cmap='gray'); axes[1].set_title('GT (grid)'); axes[1].axis('off')
            im = axes[2].imshow(sim_map, cmap='jet'); axes[2].set_title('Prototype correlation'); axes[2].axis('off')
            fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            viz_name = outdir / f"ep{ep:04d}_{mode}_q{q_sample['image_id']}.png"
            fig.suptitle(f"peak_dist={peak_to_centroid:.1f}px, iou_t0.3={ious['iou_t0.3']:.3f}")
            plt.tight_layout()
            fig.savefig(viz_name, dpi=120)
            plt.close(fig)

            results[mode].append({
                "query": q_sample['image_id'],
                "supports": [ds_train[s]['image_id'] for s in support_indices],
                "peak_to_centroid_px": peak_to_centroid,
                **ious,
                **recalls,
            })

    # Dump JSON metrics
    for mode, arr in results.items():
        out_json = outdir / f"correlation_{mode}.json"
        out_json.write_text(json.dumps(arr, indent=2), encoding='utf-8')
    print(f"Saved correlation results to {outdir}")


# ----------------- Mode: cosine -----------------

@torch.no_grad()
def run_cosine(backbone, adapter, data_root: Path, device: torch.device, outdir: Path, n_samples: int = 200, seed: int = 42):
    outdir.mkdir(parents=True, exist_ok=True)
    ds = NEUSegDataset(data_root, split="train")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n_samples, len(ds)))

    prototypes = []  # list of (name, class_id, proto)
    for idx in indices:
        sample = ds[idx]
        feat, meta = extract_adapter_feature(sample['image'], backbone, adapter, device)
        gh, gw = feat.shape[2], feat.shape[3]
        # for each foreground class present, compute prototype
        labels = np.unique(sample['masks'].numpy())
        for cls in labels:
            if int(cls) == 0:
                continue
            mask_grid = resize_mask((sample['masks'].squeeze(0)==int(cls)).float(), (gh, gw))
            if mask_grid.sum() == 0:
                continue
            proto = prototype_from_supports(feat.cpu(), mask_grid.unsqueeze(0))
            prototypes.append((sample['image_id'], int(cls), proto.numpy()))

    # compute pos / neg similarity distributions
    sims_pos = []
    sims_neg = []
    N = len(prototypes)
    for i in range(N):
        for j in range(i+1, N):
            p1 = prototypes[i][2]
            p2 = prototypes[j][2]
            cos = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-8)
            if prototypes[i][1] == prototypes[j][1]:
                sims_pos.append(cos)
            else:
                sims_neg.append(cos)

    # save hist plot and stats
    fig = plt.figure(figsize=(6,4))
    plt.hist(sims_pos, bins=50, alpha=0.7, label='positive')
    plt.hist(sims_neg, bins=50, alpha=0.7, label='negative')
    plt.legend(); plt.title('Prototype cosine similarity')
    plt.xlabel('cosine'); plt.ylabel('count')
    fig.savefig(outdir / 'cosine_proto_hist.png', dpi=120)
    plt.close(fig)

    stats = {
        'pos_mean': float(np.mean(sims_pos)) if sims_pos else None,
        'pos_std': float(np.std(sims_pos)) if sims_pos else None,
        'neg_mean': float(np.mean(sims_neg)) if sims_neg else None,
        'neg_std': float(np.std(sims_neg)) if sims_neg else None,
        'n_pos_pairs': len(sims_pos),
        'n_neg_pairs': len(sims_neg),
    }
    (outdir / 'cosine_stats.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(f"Saved cosine hist and stats to {outdir}")


# ----------------- Mode: retrieval -----------------

@torch.no_grad()
def run_retrieval(backbone, adapter, data_root: Path, device: torch.device, outdir: Path, k_list: List[int] = [1,5], seed: int = 42):
    outdir.mkdir(parents=True, exist_ok=True)
    ds_train = NEUSegDataset(data_root, split='train')
    ds_val = NEUSegDataset(data_root, split='test')

    # build prototype bank: for each train sample and for each FG class present, compute proto
    bank = []  # list of (sample_id, class, proto)
    for idx in range(len(ds_train)):
        sample = ds_train[idx]
        feat, meta = extract_adapter_feature(sample['image'], backbone, adapter, device)
        gh, gw = feat.shape[2], feat.shape[3]
        labels = np.unique(sample['masks'].numpy())
        for cls in labels:
            if int(cls) == 0:
                continue
            mask_grid = resize_mask((sample['masks'].squeeze(0)==int(cls)).float(), (gh, gw))
            if mask_grid.sum()==0:
                continue
            proto = prototype_from_supports(feat.cpu(), mask_grid.unsqueeze(0)).numpy()
            bank.append((sample['image_id'], int(cls), proto))
    bank_features = np.vstack([b[2] for b in bank])
    bank_labels = [b[1] for b in bank]

    # for each val sample and its FG classes, compute proto and retrieval rank
    recall_counts = {k: 0 for k in k_list}
    n_queries = 0
    for idx in range(len(ds_val)):
        sample = ds_val[idx]
        feat, meta = extract_adapter_feature(sample['image'], backbone, adapter, device)
        gh, gw = feat.shape[2], feat.shape[3]
        labels = np.unique(sample['masks'].numpy())
        for cls in labels:
            if int(cls) == 0:
                continue
            mask_grid = resize_mask((sample['masks'].squeeze(0)==int(cls)).float(), (gh, gw))
            if mask_grid.sum()==0:
                continue
            proto = prototype_from_supports(feat.cpu(), mask_grid.unsqueeze(0)).numpy()
            # cosine similarity to bank
            dots = bank_features @ proto
            bank_norms = np.linalg.norm(bank_features, axis=1) * (np.linalg.norm(proto) + 1e-8)
            sims = dots / (bank_norms + 1e-8)
            order = np.argsort(sims)[::-1]
            n_queries += 1
            for k in k_list:
                topk = order[:k]
                if any(bank_labels[i] == int(cls) for i in topk):
                    recall_counts[k] += 1

    recalls = {f"recall@{k}": recall_counts[k] / max(n_queries,1) for k in k_list}
    (outdir / 'retrieval.json').write_text(json.dumps({'n_queries': n_queries, **recalls}, indent=2), encoding='utf-8')
    print(f"Saved retrieval results to {outdir}; n_queries={n_queries}")


# ----------------- Mode: tsne -----------------

@torch.no_grad()
def run_tsne(backbone, adapter, data_root: Path, device: torch.device, outdir: Path, n_samples: int = 400, per_image_points: int = 50, seed: int = 42):
    if PCA is None or TSNE is None:
        print('sklearn not available. Install scikit-learn to run tsne mode.')
        return
    outdir.mkdir(parents=True, exist_ok=True)
    ds = NEUSegDataset(data_root, split='train')
    rng = random.Random(seed)
    chosen = rng.sample(range(len(ds)), min(n_samples, len(ds)))

    feats_before = []
    feats_after = []
    labels = []

    for idx in chosen:
        sample = ds[idx]
        proc, meta = preprocess_image(sample['image'])
        proc_b = proc.unsqueeze(0).to(device)
        feats_dict = backbone(proc_b)
        if isinstance(feats_dict, dict) and 'image_embedding' in feats_dict:
            emb = feats_dict['image_embedding']
        else:
            emb = feats_dict.get('stage3', list(feats_dict.values())[-1])
        # emb: [1, C, gh, gw]
        C, gh, gw = emb.shape[1], emb.shape[2], emb.shape[3]
        emb_np = emb[0].cpu().numpy()
        adapted = adapter(emb).cpu().numpy()[0]
        mask = sample['masks'].squeeze(0).numpy()
        # sample points from foreground pixels if available, else background
        ys, xs = np.where(mask != 0)
        if len(ys) == 0:
            ys, xs = np.where(np.ones_like(mask))
        # map original pixel coords to feature grid coords
        for _ in range(min(per_image_points, len(ys))):
            i = rng.randrange(len(ys))
            y, x = ys[i], xs[i]
            # rough mapping: scale to feature grid via meta.input_size -> gh
            scale_y = meta.input_size[0] / gh
            scale_x = meta.input_size[1] / gw
            gy = int(round((y / sample['image_size'][0]) * gh))
            gx = int(round((x / sample['image_size'][1]) * gw))
            gy = max(0, min(gh-1, gy)); gx = max(0, min(gw-1, gx))
            feats_before.append(emb_np[:, gy, gx])
            feats_after.append(adapted[:, gy, gx])
            labels.append(int(mask[y, x]))

    feats_before = np.vstack(feats_before)
    feats_after = np.vstack(feats_after)
    labels = np.array(labels)

    # PCA reduce then t-SNE
    pca = PCA(n_components=50)
    bf = pca.fit_transform(feats_before)
    af = pca.transform(feats_after)
    tsne = TSNE(n_components=2, init='pca', random_state=seed)
    bf2 = tsne.fit_transform(bf)
    af2 = tsne.fit_transform(af)

    def plot_emb(xy, labels, title, fname):
        plt.figure(figsize=(6,6))
        for cls in np.unique(labels):
            mask = labels==cls
            plt.scatter(xy[mask,0], xy[mask,1], s=4, label=str(cls), alpha=0.7)
        plt.legend(); plt.title(title)
        plt.savefig(outdir/fname, dpi=120); plt.close()

    plot_emb(bf2, labels, 't-SNE before adapter', 'tsne_before.png')
    plot_emb(af2, labels, 't-SNE after adapter', 'tsne_after.png')
    (outdir / 'tsne_meta.json').write_text(json.dumps({'n_points': int(len(labels))}), encoding='utf-8')
    print(f"Saved t-SNE plots to {outdir}")


# ----------------- CLI and runner -----------------

def parse_args():
    p = argparse.ArgumentParser(description='Diagnose Stage1 adapter quality for NEU_Seg')
    p.add_argument('--mode', default='correlation', choices=['correlation','cosine','retrieval','tsne','all'])
    p.add_argument('--stage1-ckpt', required=True)
    p.add_argument('--backbone-ckpt', required=True)
    p.add_argument('--data-root', default='data/NEU_Seg')
    p.add_argument('--device', default='cuda')
    p.add_argument('--outdir', default='runs/diag')
    p.add_argument('--n-episodes', type=int, default=100)
    p.add_argument('--n-samples', type=int, default=200)
    p.add_argument('--n-samples-tsne', type=int, default=400)
    p.add_argument('--n-episodes-corr', type=int, default=100)
    p.add_argument('--support-modes', default='manifest,random')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    outdir = Path(args.outdir)
    stage1_ckpt = resolve_path(args.stage1_ckpt)
    backbone_ckpt = resolve_path(args.backbone_ckpt)
    data_root = resolve_path(args.data_root)

    print(f"Loading backbone from {backbone_ckpt} on {device}")
    backbone = build_backbone(backbone_ckpt, device)
    print("Loading stage1 adapter...")
    adapter, manifest = load_stage1_adapter(stage1_ckpt, device)

    modes = [args.mode] if args.mode != 'all' else ['correlation','cosine','retrieval','tsne']
    support_modes = [m.strip() for m in args.support_modes.split(',') if m.strip()]

    if 'correlation' in modes:
        run_correlation(backbone, adapter, manifest, data_root, device, outdir / 'correlation', n_episodes=args.n_episodes, support_modes=support_modes, seed=args.seed)
    if 'cosine' in modes:
        run_cosine(backbone, adapter, data_root, device, outdir / 'cosine', n_samples=args.n_samples, seed=args.seed)
    if 'retrieval' in modes:
        run_retrieval(backbone, adapter, data_root, device, outdir / 'retrieval', k_list=[1,5,10], seed=args.seed)
    if 'tsne' in modes:
        run_tsne(backbone, adapter, data_root, device, outdir / 'tsne', n_samples=args.n_samples_tsne, per_image_points=30, seed=args.seed)

    print('Diagnostics complete.')


if __name__ == '__main__':
    main()
