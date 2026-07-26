"""
Decoder Inversion — 反向优化 Prompt，发现 Decoder 真正偏好的表示
================================================================

核心思想: 冻结 Decoder, 反向优化 dense_prompt, 看 Decoder "最希望"收到什么样的 Prompt。
然后比较 Current Prompt (PromptFusion 输出) 与 Optimal Prompt (优化得到) 的差异,
直接定位 PromptFusion 的表达瓶颈。

This directly answers: "What prompt does the decoder WANT vs what does it GET?"

实验设计:
  1. 对每个 tile, 从三种初始化出发:
     - warm_start: 从实际 dense_prompt 出发优化
     - random: 从随机噪声出发优化
     - zero: 从零出发优化
  2. 冻结 decoder (bypass_head 或 SAM MaskDecoder), 只优化 prompt
  3. 200 步 Adam 优化, 最小化 BCE + Dice loss vs GT
  4. 比较 Optimal Prompt vs Current Prompt:
     - Cosine similarity (整体 + per-channel)
     - Channel PCA / effective rank
     - Spatial frequency spectrum
     - Channel energy distribution
     - Spatial entropy
  5. 关键指标: IoU gap = Optimal IoU - Current IoU
     → 如果 gap 很大: PromptFusion 是瓶颈
     → 如果 gap 很小: Decoder 本身是瓶颈

用法:
    python tools/debug/decoder_inversion.py \
        --checkpoint <ckpt> --mode novel --k-shot 5 --num-tiles 20 --optim-steps 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adasam.adapters import CATAdapter
from adasam.backbone import build_mobile_sam, MobileSAMBackbone
from adasam.datasets.isaid_5i import ISAID5iDataset, ISAID5I_CATEGORIES
from adasam.model import AdaSAMModel, AdaSAMModelConfig
from adasam.utils.transforms import preprocess_image, resize_mask


# ═══════════════════════════════════════════════════════════════════
# Prompt Optimizer — GD on prompt with frozen decoder
# ═══════════════════════════════════════════════════════════════════

class PromptOptimizer:
    """反向优化 dense_prompt — 冻结 decoder, 只优化 prompt 最小化 loss."""

    def __init__(self, model, query_emb, sup_feat, sup_mask,
                 gt_main, H_orig, W_orig, device):
        self.model = model
        self.H = H_orig
        self.W = W_orig
        self.device = device

        # ── Detach all input tensors — they carry grad_fn from backbone/adapter
        #     and would cause "backward through the graph a second time" when
        #     used repeatedly in the 200-step optimization loop.
        self.query_emb = query_emb.detach()
        self.sup_feat = sup_feat.detach()
        self.sup_mask = sup_mask.detach()
        self.gt_main = gt_main  # [H, W] bool numpy (not a tensor)

        # Pre-compute GT at 256×256 for loss
        gt_t = torch.from_numpy(gt_main.astype(np.float32)).to(device)
        self.gt_256 = F.interpolate(
            gt_t.unsqueeze(0).unsqueeze(0), (256, 256), mode="area"
        )[0, 0]  # [256, 256]

        # Build support memory and geometric prior once (frozen)
        self.support_memory = model.support_encoder(self.sup_feat, self.sup_mask)
        self.geometric_prior = None
        if model.geometric_prior is not None:
            self.geometric_prior = model.geometric_prior(self.query_emb, self.support_memory)

        # Pre-compute support prototype once
        self.support_proto = None
        if self.model.bypass_head is None:
            self.support_proto = self.model._compute_support_prototype(
                self.sup_feat, self.sup_mask
            )

        # Disable category injection during optimization — its weight.data mutation
        # breaks the autograd graph across backward() calls.
        self._saved_cat_enabled = self.model.sam_decoder._category_enabled
        self.model.sam_decoder._category_enabled = False

    def decode(self, prompt: torch.Tensor) -> torch.Tensor:
        """Decode prompt → logits at 256×256."""
        if self.model.bypass_head is not None:
            low_res = self.model.bypass_head(prompt)
        else:
            sparse_token = prompt.mean(dim=(2, 3))
            low_res, _ = self.model.sam_decoder(
                self.query_emb, sparse_token, prompt,
                support_prototype=self.support_proto,
            )
        return low_res  # [1, 1, 256, 256]

    def restore(self):
        """Restore model state after optimization."""
        self.model.sam_decoder._category_enabled = self._saved_cat_enabled

    def loss_fn(self, logits: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """BCE + Dice loss."""
        fg = logits[0, 0]  # [256, 256]
        gt = self.gt_256  # [256, 256]

        # BCE
        bce = F.binary_cross_entropy_with_logits(fg, gt)

        # Dice
        prob = fg.sigmoid()
        inter = (prob * gt).sum()
        dice = 1.0 - (2 * inter + 1) / (prob.sum() + gt.sum() + 1)

        loss = bce + dice

        # IoU for monitoring
        pred = prob > 0.5
        iou_val = (pred & gt.bool()).sum().float() / (pred | gt.bool()).sum().float().clamp(min=1)

        return loss, {"bce": float(bce), "dice": float(dice), "iou": float(iou_val)}

    def optimize(self, init_mode: str = "warm", steps: int = 200,
                lr: float = 0.01) -> dict:
        """Optimize prompt from given initialization.

        :param init_mode: "warm" (from current dense_prompt), "random", "zero", "noise"
        :param steps: number of optimization steps
        :param lr: learning rate
        :return: dict with optimized prompt, metrics trace, final metrics
        """
        C = 256
        H, W = 64, 64

        # ── Build actual dense_prompt for comparison (no grad tracking) ──
        with torch.no_grad():
            dense_pe = self.model.sam_decoder.prompt_encoder.get_dense_pe()
            spg_out = self.model.spg(self.query_emb, self.support_memory, dense_pe)
            semantic_prior = spg_out.semantic_prior

            if self.model.prompt_fusion is not None and self.geometric_prior is not None:
                actual_prompt, _ = self.model.prompt_fusion(
                    self.geometric_prior, semantic_prior
                )
            else:
                actual_prompt = semantic_prior

            actual_prompt = actual_prompt.detach().clone()

        # ── Initialize learnable prompt (fresh leaf tensor) ──
        if init_mode == "warm":
            prompt = actual_prompt.clone()
        elif init_mode == "zero":
            prompt = torch.zeros(1, C, H, W, device=self.device)
        elif init_mode == "random":
            prompt = torch.randn(1, C, H, W, device=self.device) * 0.1
        elif init_mode == "noise":
            prompt = actual_prompt.clone()
            prompt += torch.randn_like(prompt) * prompt.std() * 0.5
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")

        prompt = prompt.detach().requires_grad_(True)
        optimizer = torch.optim.Adam([prompt], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

        trace = []
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            logits = self.decode(prompt)
            loss, metrics = self.loss_fn(logits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([prompt], 1.0)
            optimizer.step()
            scheduler.step()

            # Clean up graph references to avoid accumulation issues
            del logits

            if step % 20 == 0 or step == steps - 1:
                trace.append({
                    "step": step,
                    "loss": float(loss),
                    "iou": metrics["iou"],
                    "lr": float(scheduler.get_last_lr()[0]),
                })

        # Final decode
        with torch.no_grad():
            final_logits = self.decode(prompt)
            _, final_metrics = self.loss_fn(final_logits)

        return {
            "init_mode": init_mode,
            "optimal_prompt": prompt.detach().clone(),
            "actual_prompt": actual_prompt,
            "trace": trace,
            "final_iou": final_metrics["iou"],
            "final_loss": float(trace[-1]["loss"]) if trace else 0,
            "n_steps": steps,
        }


# ═══════════════════════════════════════════════════════════════════
# Prompt comparison metrics
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def compare_prompts(actual: torch.Tensor, optimal: torch.Tensor,
                   gt_main: np.ndarray) -> dict:
    """Compute detailed comparison between actual and optimal prompt.

    :param actual: [1, C, H, W] current PromptFusion output
    :param optimal: [1, C, H, W] optimized prompt
    :param gt_main: [H_gt, W_gt] bool GT mask
    :return: dict of comparison metrics
    """
    C, H, W = actual.shape[1], actual.shape[2], actual.shape[3]
    a = actual[0].cpu().float()  # [C, H, W]
    o = optimal[0].cpu().float()

    metrics = {}

    # 1. Cosine similarity (global)
    a_flat = a.reshape(-1)
    o_flat = o.reshape(-1)
    cos_global = float(F.cosine_similarity(a_flat.unsqueeze(0), o_flat.unsqueeze(0)))
    metrics["cos_global"] = cos_global

    # 2. Per-channel cosine similarity
    a_ch = a.reshape(C, -1)  # [C, HW]
    o_ch = o.reshape(C, -1)
    ch_cos = F.cosine_similarity(a_ch, o_ch, dim=1)  # [C]
    metrics["cos_ch_mean"] = float(ch_cos.mean())
    metrics["cos_ch_std"] = float(ch_cos.std())
    metrics["cos_ch_min"] = float(ch_cos.min())
    metrics["cos_ch_max"] = float(ch_cos.max())

    # 3. L2 distance
    metrics["l2_dist"] = float((a - o).pow(2).mean().sqrt())

    # 4. Magnitude comparison
    metrics["actual_l2_norm"] = float(a.pow(2).mean().sqrt())
    metrics["optimal_l2_norm"] = float(o.pow(2).mean().sqrt())
    metrics["magnitude_ratio"] = metrics["optimal_l2_norm"] / max(metrics["actual_l2_norm"], 1e-8)

    # 5. Channel energy distribution (Gini coefficient)
    a_energy = a.pow(2).mean(dim=(1, 2))  # [C]
    o_energy = o.pow(2).mean(dim=(1, 2))
    metrics["actual_energy_gini"] = _gini(a_energy.numpy())
    metrics["optimal_energy_gini"] = _gini(o_energy.numpy())
    metrics["actual_energy_entropy"] = float(_entropy(a_energy))
    metrics["optimal_energy_entropy"] = float(_entropy(o_energy))

    # 6. Effective rank
    metrics["actual_effective_rank"] = _effective_rank(a)
    metrics["optimal_effective_rank"] = _effective_rank(o)

    # 7. Spatial frequency (rough: variance of Laplacian)
    metrics["actual_spatial_freq"] = _spatial_frequency(a)
    metrics["optimal_spatial_freq"] = _spatial_frequency(o)

    # 8. GT-aligned Pearson r (L2 norm activation vs GT)
    gt_t = torch.from_numpy(gt_main.astype(np.float32))
    gt_rsz = F.interpolate(gt_t.unsqueeze(0).unsqueeze(0), (H, W), mode="area")[0, 0].numpy()
    gt_f = gt_rsz.ravel()

    for name, p in [("actual", a), ("optimal", o)]:
        act_l2 = p.pow(2).mean(dim=0).sqrt().numpy().ravel()
        s_a, s_g = act_l2.std(), gt_f.std()
        metrics[f"{name}_pearson_l2"] = float(np.corrcoef(act_l2, gt_f)[0, 1]) if s_a > 1e-8 and s_g > 1e-8 else 0.0

        # Best channel r
        cors = []
        for c in range(C):
            ch_f = p[c].numpy().ravel()
            s_c = ch_f.std()
            cors.append(float(np.corrcoef(ch_f, gt_f)[0, 1]) if s_c > 1e-8 and s_g > 1e-8 else 0.0)
        cors = np.array(cors)
        metrics[f"{name}_bestch_r"] = float(cors.max())
        metrics[f"{name}_n_pos_ch"] = int((cors > 0.1).sum())
        metrics[f"{name}_n_neg_ch"] = int((cors < -0.1).sum())

    # 9. Cross-channel correlation (average off-diagonal)
    metrics["actual_ch_correlation"] = _mean_ch_correlation(a)
    metrics["optimal_ch_correlation"] = _mean_ch_correlation(o)

    return metrics


def _gini(x: np.ndarray) -> float:
    """Gini coefficient of distribution (0=equal, 1=one dominates)."""
    x = np.sort(np.abs(x))
    n = len(x)
    if x.sum() < 1e-10:
        return 0.0
    return float(1 - 2 * np.sum(x * np.arange(1, n + 1)) / (n * np.sum(x)) + (n + 1) / n)


def _entropy(x: torch.Tensor) -> float:
    """Normalized entropy of distribution (0=one dominates, 1=uniform)."""
    p = x.abs() / x.abs().sum().clamp(min=1e-10)
    H = -(p * torch.log(p + 1e-10)).sum()
    return float(H / np.log(len(x)))


def _effective_rank(x: torch.Tensor) -> float:
    """Effective rank via SVD (fraction of singular values explaining 90% variance)."""
    u, s, v = torch.svd(x.reshape(x.shape[0], -1).float())
    s2 = s ** 2
    cumsum = torch.cumsum(s2, dim=0) / s2.sum()
    return float((cumsum < 0.9).sum() + 1)


def _spatial_frequency(x: torch.Tensor) -> float:
    """Mean spatial frequency (Laplacian variance) across channels."""
    laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                             dtype=torch.float32, device=x.device)
    lap = laplacian.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
    total_var = 0.0
    for c in range(min(x.shape[0], 32)):  # sample 32 channels for speed
        ch = x[c].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        lap_resp = F.conv2d(ch, lap, padding=1)
        total_var += float(lap_resp.var())
    return total_var / min(x.shape[0], 32)


def _mean_ch_correlation(x: torch.Tensor) -> float:
    """Mean absolute off-diagonal channel correlation."""
    C = x.shape[0]
    if C > 64:  # sample for speed
        idx = torch.randperm(C)[:64]
        x = x[idx]
        C = 64
    flat = x.reshape(C, -1)
    flat_n = F.normalize(flat, dim=1)
    corr = flat_n @ flat_n.T
    mask = ~torch.eye(C, dtype=torch.bool)
    return float(corr[mask].abs().mean())


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Decoder Inversion — what prompt does the decoder want?")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/iSAID-5i")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--mode", default="novel")
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-tiles", type=int, default=20)
    parser.add_argument("--optim-steps", type=int, default=200)
    parser.add_argument("--optim-lr", type=float, default=0.01)
    parser.add_argument("--init-modes", nargs="+", default=["warm", "noise"],
                       help="init modes: warm, noise, random, zero")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--force-bypass", action="store_true",
                       help="Force use BypassMaskHead even if untrained (diagnostic only)")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    fold = args.fold if args.fold is not None else ckpt.get("fold", 0)
    mode = args.mode if args.mode is not None else ckpt.get("mode", "novel")
    k_shot = args.k_shot if args.k_shot is not None else ckpt.get("k_shot", 5)

    # ── Load model ──
    bb_cfg = cfg.get("backbone", {})
    bb_path = Path(bb_cfg.get("checkpoint", "weights/mobile_sam.pt"))
    bb_path = bb_path if bb_path.is_absolute() else _REPO_ROOT / bb_path
    sam = build_mobile_sam(str(bb_path), bb_cfg.get("model_type", "vit_t"), device)
    backbone = MobileSAMBackbone(sam.image_encoder, sam.image_encoder.img_size).to(device)
    embed_dim = int(cfg.get("support_encoder", {}).get("embed_dim", 256))

    # ── Decoder selection logic ──
    # Check if checkpoint was trained with BypassMaskHead
    has_bypass_weights = any("bypass_head" in k for k in ckpt.get("model", {}))
    cfg_has_bypass = bool(cfg.get("ablation", {}).get("bypass_decoder", False))

    if has_bypass_weights and args.force_bypass:
        print("[model] Using BypassMaskHead (--force-bypass, trained weights found)")
    elif cfg_has_bypass and not has_bypass_weights:
        # Config says bypass but checkpoint has no bypass weights → untrained!
        print("=" * 70)
        print("  WARNING: Config has bypass_decoder=True but checkpoint has NO")
        print("  bypass_head weights. BypassMaskHead is UNTRAINED (random init).")
        print("  Decoder inversion requires a TRAINED decoder to be meaningful.")
        print("  → Forcing SAM Decoder instead (use --force-bypass to override).")
        print("=" * 70)
        cfg.setdefault("ablation", {})["bypass_decoder"] = False
    elif cfg_has_bypass:
        # Config says bypass but no --force-bypass → warn
        print("=" * 70)
        print("  NOTE: Config has bypass_decoder=True. Decoder inversion works")
        print("  best with the original SAM Decoder (TwoWayTransformer).")
        print("  → Using SAM Decoder (use --force-bypass for BypassMaskHead).")
        print("=" * 70)
        cfg.setdefault("ablation", {})["bypass_decoder"] = False

    model = AdaSAMModel(sam, AdaSAMModelConfig.from_dict(cfg)).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad_(False)

    cat_adapter = None
    if ckpt.get("cat_adapter") is not None:
        tcfg = cfg.get("train", {})
        cat_adapter = CATAdapter(
            dim=embed_dim,
            bottleneck=int(tcfg.get("cat_adapter", {}).get("bottleneck", 64)),
        ).to(device)
        cat_adapter.load_state_dict(ckpt["cat_adapter"]); cat_adapter.eval()

    use_sam_decoder = model.bypass_head is None
    print(f"[model] decoder={'SAM' if use_sam_decoder else 'BypassHead'}, "
          f"prompt_fusion={model.prompt_fusion is not None}")

    data_root = Path(args.data_root)
    if not data_root.is_absolute(): data_root = _REPO_ROOT / data_root
    val_ds = ISAID5iDataset(root=str(data_root), fold=fold, split="val", mode=mode)
    visible_classes = val_ds.visible_classes()

    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "decoder_inversion"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        vis_dir = out_dir / "figures"
        vis_dir.mkdir(parents=True, exist_ok=True)

    # ── Build support cache ──
    print(f"[setup] building support cache ({len(visible_classes)} classes)...")
    support_cache: dict = {}
    for cls in visible_classes:
        ds = ISAID5iDataset(root=str(data_root), fold=fold, split="train", mode=mode)
        tiles = ds.class_to_tiles(cls)
        if not tiles: continue
        scenes = defaultdict(list)
        for idx in tiles:
            src = ds._source_images.get(ds.tile_ids[idx], ds.tile_ids[idx])
            scenes[src].append(idx)
        rng = random.Random(args.seed)
        keys = list(scenes)
        k = min(k_shot, len(keys))
        chosen = rng.sample(keys, k)
        images, masks = [], []
        for sid in chosen:
            idx = rng.choice(scenes[sid])
            s = ds[idx]
            fg = ds.get_class_mask(idx, cls)
            if fg is None or fg.sum() < 1: continue
            xx, _ = preprocess_image(s["image"])
            images.append(xx.to(device)); masks.append(fg)
        if not images: continue
        feats = backbone(torch.stack(images, dim=0))["image_embedding"]
        if cat_adapter is not None: feats = cat_adapter(feats)
        mg = torch.stack([resize_mask(m, (feats.shape[2], feats.shape[3])).to(device)
                          for m in masks], dim=0)
        support_cache[cls] = (feats, mg)
    print(f"[setup] support cache: {len(support_cache)}/{len(visible_classes)}")

    # ── Find tiles ──
    candidates = []
    for idx in range(len(val_ds)):
        present = [c for c in visible_classes
                   if val_ds.get_class_mask(idx, c) is not None
                   and val_ds.get_class_mask(idx, c).sum() > 50]
        if len(present) >= 1: candidates.append((idx, present))
    random.Random(args.seed + 9999).shuffle(candidates)
    selected = candidates[:args.num_tiles]
    print(f"[select] {len(candidates)} candidates, using {len(selected)}")

    # ── Run inversion ──
    all_results = {mode: [] for mode in args.init_modes}  # per-init-mode results
    per_tile_summary = []

    for tile_idx, present_classes in tqdm(selected, desc="inversion"):
        sample = val_ds[tile_idx]
        H, W = sample["image"].shape[1], sample["image"].shape[2]
        img_np = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        xx, meta = preprocess_image(sample["image"])
        query_emb = backbone(xx.unsqueeze(0).to(device))["image_embedding"]
        if cat_adapter is not None: query_emb = cat_adapter(query_emb)

        main_cls = max(present_classes, key=lambda c: val_ds.get_class_mask(tile_idx, c).sum())
        sup_data = support_cache.get(main_cls)
        if sup_data is None: continue
        sup_feat, sup_mask = sup_data
        gt_main = val_ds.get_class_mask(tile_idx, main_cls).numpy().astype(bool)

        opt = PromptOptimizer(model, query_emb, sup_feat, sup_mask,
                             gt_main, H, W, device)

        tile_results = {}
        for init_mode in args.init_modes:
            result = opt.optimize(
                init_mode=init_mode, steps=args.optim_steps, lr=args.optim_lr
            )
            tile_results[init_mode] = result

        # Compare actual vs optimal (warm start)
        warm_result = tile_results.get("warm")
        if warm_result is not None:
            cmp = compare_prompts(
                warm_result["actual_prompt"],
                warm_result["optimal_prompt"],
                gt_main,
            )
            cmp["tile_idx"] = tile_idx
            cmp["class_id"] = main_cls
            cmp["class_name"] = ISAID5I_CATEGORIES.get(main_cls, f"cls{main_cls}")
            cmp["tile_id"] = sample.get("tile_id", str(tile_idx))
            cmp["actual_iou"] = _eval_iou(model, warm_result["actual_prompt"],
                                         query_emb, sup_feat, sup_mask, gt_main, H, W)
            cmp["optimal_iou"] = warm_result["final_iou"]
            cmp["iou_gap"] = cmp["optimal_iou"] - cmp["actual_iou"]
            all_results["warm"].append(cmp)

        # Per-tile summary
        summary_entry = {
            "tile_idx": tile_idx, "class_id": main_cls,
            "class_name": ISAID5I_CATEGORIES.get(main_cls, f"cls{main_cls}"),
        }
        for init_mode in args.init_modes:
            r = tile_results[init_mode]
            summary_entry[f"{init_mode}_init_iou"] = r["trace"][0]["iou"] if r["trace"] else 0
            summary_entry[f"{init_mode}_final_iou"] = r["final_iou"]
        if "warm" in tile_results:
            summary_entry["actual_iou"] = cmp["actual_iou"]
            summary_entry["iou_gap"] = cmp["iou_gap"]
        per_tile_summary.append(summary_entry)

        # Visualization (first 5 tiles)
        if args.save_vis and len(per_tile_summary) <= 5:
            _save_tile_figure(img_np, gt_main, H, W, tile_results,
                            sample.get("tile_id", str(tile_idx)), main_cls, vis_dir)

    # ── Aggregate ──
    summary = _aggregate_results(all_results, per_tile_summary)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    per_tile_path = out_dir / "per_tile.json"
    with open(per_tile_path, "w") as f:
        json.dump(per_tile_summary, f, indent=2, ensure_ascii=False)

    # ── Print report ──
    _print_inversion_report(summary, per_tile_summary)

    if args.save_vis:
        _save_summary_charts(summary, per_tile_summary, all_results, vis_dir)

    print(f"\n[Summary] {summary_path}")
    print(f"[Per-tile] {per_tile_path}")
    print("[Done]")


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def _eval_iou(model, prompt, query_emb, sup_feat, sup_mask, gt_main, H, W):
    """Evaluate IoU of a prompt through the decoder."""
    if model.bypass_head is not None:
        low_res = model.bypass_head(prompt)
    else:
        sparse_token = prompt.mean(dim=(2, 3))
        support_proto = model._compute_support_prototype(sup_feat, sup_mask)
        low_res, _ = model.sam_decoder(query_emb, sparse_token, prompt,
                                       support_prototype=support_proto)
    pred_logits = F.interpolate(low_res.float(), size=(H, W),
                                mode="bilinear", align_corners=False)[0, 0]
    vals = pred_logits.cpu()
    if vals.min() >= 0 and vals.max() <= 1:
        pred = vals > 0.5
    else:
        pred = vals.sigmoid() > 0.5
    pred_np = pred.numpy()
    inter = (pred_np & gt_main).sum()
    union = (pred_np | gt_main).sum()
    return float(inter / union) if union > 0 else 0.0


def _aggregate_results(all_results: dict, per_tile_summary: list) -> dict:
    """Aggregate comparison metrics across tiles."""
    summary = {}

    # Comparison metrics (only for warm start)
    if "warm" in all_results and all_results["warm"]:
        items = all_results["warm"]
        sm = {}
        numeric_keys = [k for k in items[0] if isinstance(items[0][k], (int, float, np.floating))
                       and k not in ("tile_idx", "class_id")]
        for key in numeric_keys:
            vals = [d[key] for d in items if d.get(key) is not None]
            if vals:
                sm[key] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "median": round(float(np.median(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                    "min": round(float(np.min(vals)), 4),
                    "max": round(float(np.max(vals)), 4),
                }
        summary["comparison"] = {"n_tiles": len(items), "metrics": sm}

    # Per-init IoU summary
    if per_tile_summary:
        init_summary = {}
        for key in per_tile_summary[0]:
            if "_iou" in key:
                vals = [d[key] for d in per_tile_summary if d.get(key) is not None]
                if vals:
                    init_summary[key] = {
                        "mean": round(float(np.mean(vals)), 4),
                        "median": round(float(np.median(vals)), 4),
                        "std": round(float(np.std(vals)), 4),
                    }
        summary["iou_by_init"] = init_summary

    return summary


def _print_inversion_report(summary: dict, per_tile_summary: list):
    SEP = "=" * 90
    print(f"\n{SEP}")
    print("  Decoder Inversion — What prompt does the decoder actually want?")
    print(f"{SEP}")

    # IoU gap
    if "iou_by_init" in summary:
        print(f"\n  ┌─ IoU by Initialization{'─'*66}")
        ios = summary["iou_by_init"]
        for key in sorted(ios.keys()):
            m = ios[key]
            label = key.replace("_iou", "").replace("_", " ")
            print(f"  {label:<25s}  IoU={m['mean']:.4f} ± {m['std']:.4f}")

    # Comparison metrics
    if "comparison" in summary:
        cmp = summary["comparison"]["metrics"]
        print(f"\n  ┌─ Key Comparison: Current Prompt vs Optimal Prompt (warm start){'─'*30}")

        # Critical gap
        iou_gap = cmp.get("iou_gap", {})
        actual_iou = cmp.get("actual_iou", {})
        optimal_iou = cmp.get("optimal_iou", {})
        if iou_gap.get("mean") is not None:
            print(f"\n  ★ IoU GAP (how much can prompt optimization improve?)")
            print(f"    Current (PromptFusion):  {actual_iou.get('mean', 0):.4f}")
            print(f"    Optimal (200-step GD):   {optimal_iou.get('mean', 0):.4f}")
            print(f"    GAP:                      {iou_gap.get('mean', 0):+.4f}")
            gap = iou_gap.get("mean", 0)
            if gap > 0.1:
                print(f"    → LARGE GAP: PromptFusion IS the bottleneck. Decoder can do much better.")
            elif gap > 0.03:
                print(f"    → MODERATE GAP: PromptFusion has room for improvement.")
            else:
                print(f"    → SMALL GAP: Decoder itself is the bottleneck, not PromptFusion.")

        # Distribution comparison
        print(f"\n  [Prompt Distribution Mismatch]")
        for key, label in [
            ("cos_global", "Global cosine similarity"),
            ("cos_ch_mean", "Mean per-channel cosine"),
            ("magnitude_ratio", "Magnitude ratio (opt/act)"),
            ("l2_dist", "L2 distance"),
        ]:
            if key in cmp:
                print(f"    {label:<30s} {cmp[key]['mean']:.4f}")

        print(f"\n  [Channel Structure]")
        for key, label in [
            ("actual_effective_rank", "Actual eff. rank"),
            ("optimal_effective_rank", "Optimal eff. rank"),
            ("actual_energy_gini", "Actual energy Gini"),
            ("optimal_energy_gini", "Optimal energy Gini"),
            ("actual_energy_entropy", "Actual energy entropy"),
            ("optimal_energy_entropy", "Optimal energy entropy"),
        ]:
            if key in cmp:
                print(f"    {label:<30s} {cmp[key]['mean']:.4f}")

        print(f"\n  [Spatial Structure]")
        for key, label in [
            ("actual_spatial_freq", "Actual spatial freq"),
            ("optimal_spatial_freq", "Optimal spatial freq"),
            ("actual_pearson_l2", "Actual Pearson r (L2)"),
            ("optimal_pearson_l2", "Optimal Pearson r (L2)"),
            ("actual_bestch_r", "Actual best-ch r"),
            ("optimal_bestch_r", "Optimal best-ch r"),
        ]:
            if key in cmp:
                print(f"    {label:<30s} {cmp[key]['mean']:.4f}")

        print(f"\n  [Channel Counts]")
        for key, label in [
            ("actual_n_pos_ch", "Actual n_pos_ch"),
            ("optimal_n_pos_ch", "Optimal n_pos_ch"),
            ("actual_n_neg_ch", "Actual n_neg_ch"),
            ("optimal_n_neg_ch", "Optimal n_neg_ch"),
            ("actual_ch_correlation", "Actual ch correlation"),
            ("optimal_ch_correlation", "Optimal ch correlation"),
        ]:
            if key in cmp:
                print(f"    {label:<30s} {cmp[key]['mean']:.4f}")

    # Per-tile winners
    if per_tile_summary:
        warm_improved = 0
        for e in per_tile_summary:
            gap = e.get("iou_gap", 0) or 0
            if gap > 0.01:
                warm_improved += 1
        n = len(per_tile_summary)
        print(f"\n  [Per-Tile] Warm-start optimization improved IoU on {warm_improved}/{n} tiles "
              f"({100*warm_improved/max(n,1):.0f}%)")

    print(f"\n{SEP}")


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

def _save_tile_figure(img_np, gt_main, H, W, tile_results, tile_id, main_cls, vis_dir):
    cls_name = ISAID5I_CATEGORIES.get(main_cls, f"cls{main_cls}")
    init_modes = list(tile_results.keys())
    n_modes = len(init_modes)
    fig, axes = plt.subplots(3, max(n_modes + 1, 4), figsize=(4 * (n_modes + 1), 10))

    # Row 1: Query, GT, and per-init optimal prompt → prediction
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Query", fontsize=8); axes[0, 0].axis("off")
    axes[0, 1].imshow(gt_main, cmap="gray")
    axes[0, 1].set_title(f"GT ({cls_name})", fontsize=8); axes[0, 1].axis("off")

    for i, init_mode in enumerate(init_modes):
        r = tile_results[init_mode]
        opt_prompt = r["optimal_prompt"]
        model = None  # We don't have model here, skip for now — will use result trace
        ax = axes[0, i + 2]
        ax.text(0.5, 0.5, f"{init_mode}\nIoU={r['final_iou']:.3f}",
               ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title(f"Opt ({init_mode})", fontsize=8); ax.axis("off")

    # Row 2: Actual prompt L2 norm and per-init optimal prompt L2 norm
    warm = tile_results.get("warm")
    if warm is not None:
        a_l2 = warm["actual_prompt"][0].pow(2).mean(dim=0).sqrt().cpu().numpy()
        im = axes[1, 0].imshow(a_l2, cmap="RdBu_r")
        axes[1, 0].set_title("Actual DP L2", fontsize=8); axes[1, 0].axis("off")
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

        o_l2 = warm["optimal_prompt"][0].pow(2).mean(dim=0).sqrt().cpu().numpy()
        im = axes[1, 1].imshow(o_l2, cmap="RdBu_r")
        axes[1, 1].set_title("Optimal DP L2", fontsize=8); axes[1, 1].axis("off")
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

        # Difference
        diff_l2 = np.abs(o_l2 - a_l2)
        im = axes[1, 2].imshow(diff_l2, cmap="hot")
        axes[1, 2].set_title("|Opt - Act| L2", fontsize=8); axes[1, 2].axis("off")
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

    # Row 3: Optimization traces
    for i, init_mode in enumerate(init_modes):
        r = tile_results[init_mode]
        trace = r["trace"]
        ax = axes[2, i]
        steps = [t["step"] for t in trace]
        ious = [t["iou"] for t in trace]
        ax.plot(steps, ious, "o-", markersize=2)
        ax.set_xlabel("Step"); ax.set_ylabel("IoU")
        ax.set_title(f"{init_mode} trace", fontsize=8)
        ax.set_ylim(0, 1)
        if warm is not None:
            ax.axhline(y=warm.get("final_iou", 0) if init_mode == "warm" else
                         tile_results.get("warm", {}).get("final_iou", 0),
                      color="green" if init_mode == "warm" else "gray",
                      linestyle="--", alpha=0.5)

    # Extra columns
    for j in range(i + 1, axes.shape[1]):
        axes[2, j].axis("off")

    fig.suptitle(f"Decoder Inversion: {tile_id} | support={cls_name}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(vis_dir / f"{tile_id}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_summary_charts(summary, per_tile_summary, all_results, vis_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Chart 1: IoU gap histogram
    ax = axes[0, 0]
    gaps = [e.get("iou_gap", 0) or 0 for e in per_tile_summary]
    ax.hist(gaps, bins=20, color="steelblue", edgecolor="white")
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    ax.axvline(x=np.mean(gaps), color="green", linestyle="-", alpha=0.7,
              label=f"mean={np.mean(gaps):+.4f}")
    ax.set_xlabel("IoU Gap (Optimal - Current)")
    ax.set_ylabel("N tiles")
    ax.set_title("IoU Improvement from Prompt Optimization", fontsize=10, fontweight="bold")
    ax.legend()

    # Chart 2: Current vs Optimal IoU scatter
    ax = axes[0, 1]
    actuals = [e.get("actual_iou", 0) or 0 for e in per_tile_summary]
    optimals = [e.get("warm_final_iou", 0) or 0 for e in per_tile_summary]
    if actuals and optimals:
        max_v = max(max(actuals), max(optimals)) * 1.1
        ax.scatter(actuals, optimals, alpha=0.6, s=15, c="steelblue")
        ax.plot([0, max_v], [0, max_v], "k--", alpha=0.3)
        above = sum(1 for a, o in zip(actuals, optimals) if o > a)
        ax.set_xlabel("Current IoU"); ax.set_ylabel("Optimal IoU (warm)")
        ax.set_title(f"Current vs Optimal IoU\n(opt>{above}/{len(actuals)})",
                    fontsize=10, fontweight="bold")

    # Chart 3: Comparison metric bars
    ax = axes[0, 2]
    if "comparison" in summary:
        cmp = summary["comparison"]["metrics"]
        bar_metrics = [
            ("cos_global", "Cos\n(global)"),
            ("cos_ch_mean", "Cos\n(per-ch)"),
            ("magnitude_ratio", "Mag\nratio"),
            ("actual_effective_rank", "Eff rank\n(act)"),
            ("optimal_effective_rank", "Eff rank\n(opt)"),
            ("actual_spatial_freq", "Spat freq\n(act)"),
            ("optimal_spatial_freq", "Spat freq\n(opt)"),
        ]
        means, labels, colors = [], [], []
        for key, label in bar_metrics:
            if key in cmp and cmp[key]["mean"] is not None:
                means.append(cmp[key]["mean"])
                labels.append(label)
                colors.append("steelblue" if "actual" in key else "coral" if "optimal" in key else "gray")
        ax.bar(range(len(means)), means, color=colors, edgecolor="white")
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title("Prompt Comparison Metrics", fontsize=10, fontweight="bold")

    # Chart 4: Channel energy Gini comparison
    ax = axes[1, 0]
    if "comparison" in summary and "warm" in all_results:
        items = all_results["warm"]
        act_gini = [d["actual_energy_gini"] for d in items]
        opt_gini = [d["optimal_energy_gini"] for d in items]
        ax.scatter(act_gini, opt_gini, alpha=0.6, s=15, c="steelblue")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("Actual energy Gini"); ax.set_ylabel("Optimal energy Gini")
        ax.set_title("Channel Energy Concentration", fontsize=10, fontweight="bold")

    # Chart 5: Optimization convergence
    ax = axes[1, 1]
    if per_tile_summary:
        for init_mode in ["warm", "noise"]:
            key = f"{init_mode}_init_iou"
            val_key = f"{init_mode}_final_iou"
            inits = [e.get(key, 0) or 0 for e in per_tile_summary if key in e]
            finals = [e.get(val_key, 0) or 0 for e in per_tile_summary if val_key in e]
            if inits and finals:
                ax.scatter(inits, finals, alpha=0.6, s=15, label=init_mode)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("Initial IoU"); ax.set_ylabel("Final IoU")
        ax.set_title("Optimization Convergence", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)

    # Chart 6: Per-tile IoU gap ranked
    ax = axes[1, 2]
    gaps_sorted = sorted(gaps, reverse=True)
    colors_gap = ["#2ecc71" if g > 0 else "#e74c3c" for g in gaps_sorted]
    ax.bar(range(len(gaps_sorted)), gaps_sorted, color=colors_gap, edgecolor="white")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Tile (ranked)"); ax.set_ylabel("IoU Gap")
    ax.set_title("Per-Tile IoU Gap (ranked)", fontsize=10, fontweight="bold")
    n_pos = sum(1 for g in gaps_sorted if g > 0.01)
    ax.text(0.95, 0.95, f"Improved: {n_pos}/{len(gaps_sorted)}",
           transform=ax.transAxes, fontsize=9, ha="right", va="top",
           bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle("Decoder Inversion Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(vis_dir / "summary_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
