"""
Decoder 可解释性实验 — 共享基础模块
=====================================

提供所有 decoder interpretability 工具共用的 boilerplate:
  - 模型加载 (MobileSAM backbone + CATAdapter + AdaSAMModel)
  - Support cache 构建
  - Multi-class tile 选择
  - 统一输出格式 (summary.json + per_tile.json)
  - 诊断报告打印

使用方式: 继承 DecoderDiagBase 或调用 build_diag_context().
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adasam.adapters import CATAdapter
from adasam.backbone import build_mobile_sam, MobileSAMBackbone
from adasam.datasets.isaid_5i import ISAID5iDataset, ISAID5I_CATEGORIES
from adasam.model import AdaSAMModel, AdaSAMModelConfig
from adasam.utils import set_seed
from adasam.utils.transforms import preprocess_image, resize_mask


# ═══════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════


@dataclass
class DiagContext:
    """Loaded model + data context for a diagnostic run."""

    device: torch.device
    seed: int
    backbone: MobileSAMBackbone
    adapter: CATAdapter | None
    model: AdaSAMModel
    embed_dim: int

    # Dataset
    dataset: ISAID5iDataset
    data_root: Path
    fold: int
    mode: str  # "base" | "novel"
    k_shot: int
    visible_classes: list[int]

    # Support cache: {class_id: (features [K,C,H,W], masks [K,H,W])}
    support_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)

    # Selected tiles: [(tile_idx, [class_ids_present])]
    selected_tiles: list[tuple[int, list[int]]] = field(default_factory=list)

    # Output
    out_dir: Path | None = None

    @property
    def num_tiles(self) -> int:
        return len(self.selected_tiles)


# ═══════════════════════════════════════════════════════════════════
# Core builder
# ═══════════════════════════════════════════════════════════════════


def build_diag_context(
    args: argparse.Namespace,
    *,
    require_adapter: bool = True,
    split: str = "val",
) -> DiagContext:
    """Load model + data and build support cache.

    This is the single entry point for all decoder diagnostic tools.
    Call once, get everything needed.

    :param args: parsed CLI args (must have: stage2_ckpt, data_root, fold,
                 k_shot, seed; optionally: num_tiles, mode, output_dir)
    :param require_adapter: if True, error when adapter is missing.
    :param split: dataset split ("val" or "train").
    """
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} seed={args.seed}")

    # ── Load checkpoint ──
    ckpt_path = Path(args.stage2_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    fold = args.fold if args.fold is not None else ckpt.get("fold", 0)
    mode = args.mode if hasattr(args, "mode") and args.mode else "novel"
    k_shot = args.k_shot if args.k_shot is not None else ckpt.get("k_shot", 5)
    embed_dim = int(cfg.get("support_encoder", {}).get("embed_dim", 256))

    print(f"[setup] fold={fold} mode={mode} k_shot={k_shot}")

    # ── Backbone ──
    bb_cfg = cfg.get("backbone", {})
    bb_path = Path(bb_cfg.get("checkpoint", "weights/mobile_sam.pt"))
    bb_path = bb_path if bb_path.is_absolute() else _REPO_ROOT / bb_path
    sam = build_mobile_sam(str(bb_path), bb_cfg.get("model_type", "vit_t"), device)
    backbone = MobileSAMBackbone(sam.image_encoder, sam.image_encoder.img_size).to(device)

    # ── Model ──
    # Decoder selection: prefer SAM Decoder for interpretability experiments.
    # If the checkpoint was trained with bypass_decoder=True, BypassMaskHead
    # may be untrained (random init). Force SAM Decoder unless the checkpoint
    # actually has trained bypass_head weights.
    cfg_has_bypass = bool(cfg.get("ablation", {}).get("bypass_decoder", False))
    has_bypass_weights = any("bypass_head" in k for k in ckpt.get("model", {}))
    if cfg_has_bypass and not has_bypass_weights:
        print("=" * 70)
        print("  NOTE: Config has bypass_decoder=True but checkpoint has NO")
        print("  bypass_head weights. Forcing SAM Decoder instead.")
        print("=" * 70)
        cfg.setdefault("ablation", {})["bypass_decoder"] = False
    elif cfg_has_bypass:
        print("=" * 70)
        print("  NOTE: Config has bypass_decoder=True. Diagnostic tools work")
        print("  best with the original SAM Decoder (TwoWayTransformer).")
        print("  → Forcing SAM Decoder for this run.")
        print("=" * 70)
        cfg.setdefault("ablation", {})["bypass_decoder"] = False

    model_cfg = AdaSAMModelConfig.from_dict(cfg)
    model = AdaSAMModel(sam, model_cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    use_sam_decoder = model.bypass_head is None
    print(f"[model] decoder={'SAM' if use_sam_decoder else 'BypassHead'}, "
          f"prompt_fusion={model.prompt_fusion is not None}")

    # ── Adapter ──
    adapter = None
    adapter_state = ckpt.get("cat_adapter")
    if adapter_state is not None:
        tcfg = cfg.get("train", {})
        adapter = CATAdapter(
            dim=embed_dim,
            bottleneck=int(tcfg.get("cat_adapter", {}).get("bottleneck", 64)),
        ).to(device)
        adapter.load_state_dict(adapter_state)
        adapter.eval()
        for p in adapter.parameters():
            p.requires_grad_(False)
        print(f"[setup] adapter loaded from checkpoint")
    elif require_adapter:
        raise RuntimeError(
            "Checkpoint has no adapter weights. "
            "Train with Stage 1 first, or set require_adapter=False."
        )
    else:
        print("[setup] NO adapter (raw SAM features)")

    # ── Dataset ──
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = _REPO_ROOT / data_root
    dataset = ISAID5iDataset(root=str(data_root), fold=fold, split=split, mode=mode)
    visible_classes = dataset.visible_classes()
    print(f"[setup] dataset: {len(dataset)} tiles, {len(visible_classes)} visible classes")

    ctx = DiagContext(
        device=device,
        seed=args.seed,
        backbone=backbone,
        adapter=adapter,
        model=model,
        embed_dim=embed_dim,
        dataset=dataset,
        data_root=data_root,
        fold=fold,
        mode=mode,
        k_shot=k_shot,
        visible_classes=visible_classes,
    )

    # ── Output dir ──
    out_dir = getattr(args, "output_dir", None)
    if out_dir:
        ctx.out_dir = Path(out_dir)
    else:
        ctx.out_dir = ckpt_path.parent / "decoder_diag"
    ctx.out_dir.mkdir(parents=True, exist_ok=True)

    return ctx


# ═══════════════════════════════════════════════════════════════════
# Support cache
# ═══════════════════════════════════════════════════════════════════


def build_support_cache(ctx: DiagContext) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Build per-class support cache for the visible classes.

    :return: {class_id: (features [K, C, gh, gw], masks [K, gh, gw])}
    """
    rng = random.Random(ctx.seed)
    train_ds = ISAID5iDataset(
        root=str(ctx.data_root), fold=ctx.fold, split="train", mode=ctx.mode,
    )
    cache: dict = {}
    for cls in ctx.visible_classes:
        tiles = train_ds.class_to_tiles(cls)
        if not tiles:
            continue
        # Group by source image for scene-disjoint sampling
        scenes = defaultdict(list)
        for idx in tiles:
            src = train_ds._source_images.get(train_ds.tile_ids[idx], train_ds.tile_ids[idx])
            scenes[src].append(idx)
        keys = list(scenes.keys())
        k = min(ctx.k_shot, len(keys))
        chosen = rng.sample(keys, k)
        images, masks = [], []
        for sid in chosen:
            idx = rng.choice(scenes[sid])
            s = train_ds[idx]
            fg = train_ds.get_class_mask(idx, cls)
            if fg is None or fg.sum() < 1:
                continue
            xx, _ = preprocess_image(s["image"])
            images.append(xx.to(ctx.device))
            masks.append(fg)
        if not images:
            continue
        feats = ctx.backbone(torch.stack(images, dim=0))["image_embedding"]
        if ctx.adapter is not None:
            feats = ctx.adapter(feats)
        mg = torch.stack([
            resize_mask(m, (feats.shape[2], feats.shape[3])).to(ctx.device)
            for m in masks
        ], dim=0)
        cache[cls] = (feats, mg)
    ctx.support_cache = cache
    print(f"[setup] support cache: {len(cache)}/{len(ctx.visible_classes)} classes")
    return cache


# ═══════════════════════════════════════════════════════════════════
# Tile selection
# ═══════════════════════════════════════════════════════════════════


def select_tiles(
    ctx: DiagContext,
    num_tiles: int = 20,
    min_pixels: int = 100,
    min_classes: int = 1,
) -> list[tuple[int, list[int]]]:
    """Select multi-class tiles for evaluation.

    :param ctx: diagnostic context with dataset loaded.
    :param num_tiles: maximum number of tiles to select.
    :param min_pixels: minimum FG pixels per class to include.
    :param min_classes: minimum number of visible classes per tile.
    :return: list of (tile_idx, [class_ids_present]).
    """
    candidates = []
    for idx in range(len(ctx.dataset)):
        present = [
            c for c in ctx.visible_classes
            if ctx.dataset.get_class_mask(idx, c) is not None
            and ctx.dataset.get_class_mask(idx, c).sum() > min_pixels
        ]
        if len(present) >= min_classes:
            candidates.append((idx, present))
    rng = random.Random(ctx.seed + 9999)
    rng.shuffle(candidates)
    selected = candidates[:num_tiles]
    ctx.selected_tiles = selected
    print(f"[select] {len(candidates)} candidates, using {len(selected)}")
    return selected


# ═══════════════════════════════════════════════════════════════════
# Dense prompt extraction
# ═══════════════════════════════════════════════════════════════════


@torch.no_grad()
def extract_dense_prompt(
    ctx: DiagContext,
    query_emb: torch.Tensor,
    sup_feat: torch.Tensor,
    sup_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run SupportEncoder + SPG + PromptFusion → dense_prompt.

    :param query_emb: [1, C, gh, gw] query features.
    :param sup_feat: [K, C, gh, gw] support features.
    :param sup_mask: [K, gh, gw] support FG masks.
    :return: (dense_prompt [1,C,gh,gw], semantic_prior [1,C,gh,gw],
              geometric_prior [1,C,gh,gw] or None)
    """
    model = ctx.model
    support_memory = model.support_encoder(sup_feat, sup_mask)

    geometric_prior = None
    if model.geometric_prior is not None:
        geometric_prior = model.geometric_prior(query_emb, support_memory)

    dense_pe = model.sam_decoder.prompt_encoder.get_dense_pe()
    spg_out = model.spg(query_emb, support_memory, dense_pe)
    semantic_prior = spg_out.semantic_prior

    if model.prompt_fusion is not None and geometric_prior is not None:
        dense_prompt, _ = model.prompt_fusion(geometric_prior, semantic_prior)
    else:
        dense_prompt = model._build_dense_prompt(
            support_memory, sup_feat, sup_mask
        )
        if dense_prompt is None:
            dense_prompt = semantic_prior

    return dense_prompt, semantic_prior, geometric_prior


# ═══════════════════════════════════════════════════════════════════
# Quick IoU evaluation
# ═══════════════════════════════════════════════════════════════════


@torch.no_grad()
def eval_iou(
    model: AdaSAMModel,
    prompt: torch.Tensor,
    query_emb: torch.Tensor,
    sup_feat: torch.Tensor,
    sup_mask: torch.Tensor,
    gt: np.ndarray,
    H: int,
    W: int,
) -> float:
    """Evaluate IoU of a dense_prompt through the decoder.

    :param prompt: [1, C, gh, gw] dense prompt.
    :param gt: [H, W] boolean GT mask.
    :return: IoU (0-1).
    """
    if model.bypass_head is not None:
        low_res = model.bypass_head(prompt)
    else:
        sparse_token = prompt.mean(dim=(2, 3))
        support_proto = model._compute_support_prototype(sup_feat, sup_mask)
        low_res, _ = model.sam_decoder(
            query_emb, sparse_token, prompt,
            support_prototype=support_proto,
        )
    pred_logits = F.interpolate(
        low_res.float(), size=(H, W), mode="bilinear", align_corners=False,
    )[0, 0]
    vals = pred_logits.cpu()
    if vals.min() >= 0 and vals.max() <= 1:
        pred = vals > 0.5
    else:
        pred = vals.sigmoid() > 0.5
    pred_np = pred.numpy()
    inter = (pred_np & gt).sum()
    union = (pred_np | gt).sum()
    return float(inter / union) if union > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════
# Output helpers
# ═══════════════════════════════════════════════════════════════════


def save_results(
    out_dir: Path,
    summary: dict,
    per_tile: list[dict],
) -> tuple[Path, Path]:
    """Save summary.json and per_tile.json. Returns paths."""
    s_path = out_dir / "summary.json"
    with open(s_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    p_path = out_dir / "per_tile.json"
    with open(p_path, "w") as f:
        json.dump(per_tile, f, indent=2, ensure_ascii=False, default=str)

    return s_path, p_path


def aggregate_metrics(
    per_tile: list[dict],
    metric_keys: list[str] | None = None,
) -> dict:
    """Aggregate per-tile metrics → mean/median/std/min/max.

    :param per_tile: list of per-tile dicts.
    :param metric_keys: keys to aggregate (auto-detect numeric if None).
    """
    if metric_keys is None and per_tile:
        metric_keys = [
            k for k, v in per_tile[0].items()
            if isinstance(v, (int, float, np.floating))
        ]
    agg = {}
    for key in (metric_keys or []):
        vals = [d[key] for d in per_tile if key in d and d[key] is not None]
        if not vals:
            continue
        vals_arr = np.array([float(v) for v in vals])
        agg[key] = {
            "mean": round(float(vals_arr.mean()), 4),
            "median": round(float(np.median(vals_arr)), 4),
            "std": round(float(vals_arr.std()), 4),
            "min": round(float(vals_arr.min()), 4),
            "max": round(float(vals_arr.max()), 4),
            "n": len(vals),
        }
    return agg


def print_header(title: str, width: int = 90):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_metric_table(metrics: dict, indent: int = 4):
    """Print aggregated metrics as a formatted table."""
    prefix = " " * indent
    for key, stats in sorted(metrics.items()):
        if isinstance(stats, dict) and "mean" in stats:
            print(f"{prefix}{key:<35s} {stats['mean']:.4f} ± {stats['std']:.4f}  "
                  f"(n={stats.get('n', '?')})")


# ═══════════════════════════════════════════════════════════════════
# Common CLI
# ═══════════════════════════════════════════════════════════════════


def add_common_args(parser: argparse.ArgumentParser):
    """Add standard CLI arguments shared by all decoder diagnostic tools."""
    parser.add_argument("--stage2-ckpt", required=True,
                        help="Path to Stage 2 checkpoint (best_model.pt)")
    parser.add_argument("--data-root", default="data/iSAID-5i")
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold (auto-detected from checkpoint if omitted)")
    parser.add_argument("--mode", default="novel", choices=["base", "novel"])
    parser.add_argument("--k-shot", type=int, default=None,
                        help="K-shot (auto-detected from checkpoint if omitted)")
    parser.add_argument("--num-tiles", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory")
    parser.add_argument("--save-vis", action="store_true",
                        help="Save visualization figures")
    parser.add_argument("--device", default=None)
