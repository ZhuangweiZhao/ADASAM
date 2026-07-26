"""
Attention Rollout — Where Does the TwoWayTransformer Look?
===========================================================

核心问题: TwoWayTransformer 的 cross-attention 在关注 prompt/image 的哪些区域?

方法:
  1. 在 TwoWayTransformer 的每个 Attention 层注册 forward hook
  2. 提取 attention weights [B, N_heads, N_q, N_k]
  3. Attention Rollout: 递归相乘各层 attention matrix (含残差连接)
     rollout^{(l)} = A^{(l)} @ rollout^{(l-1)}  (normalized)
  4. 对 image tokens (src) 的 attention: 映射回 64×64 spatial grid
  5. 对 token (sparse) 的 self-attention: token-to-token 交互

输出:
  - Per-layer attention rollout heatmap (64×64 spatial, 6×6 token)
  - Token-to-image cross-attention: spatial focus map
  - Image-to-token cross-attention: 哪些 image region 被 tokens 关注
  - Rollout map 与 GT mask 的 IoU

用法:
    python tools/analysis/diag_attention_rollout.py \
        --stage2-ckpt runs/stage2_fold1_k5_seed42/best_model.pt \
        --data-root data/iSAID-5i --fold 1 --mode novel --k-shot 5 \
        --num-tiles 20
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
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adasam.datasets.isaid_5i import ISAID5I_CATEGORIES
from adasam.utils.transforms import preprocess_image

from tools.analysis._decoder_diag_base import (
    DiagContext,
    add_common_args,
    build_diag_context,
    build_support_cache,
    select_tiles,
    extract_dense_prompt,
    print_header,
    print_metric_table,
    aggregate_metrics,
    save_results,
)


# ═══════════════════════════════════════════════════════════════════
# Attention Hook + Rollout
# ═══════════════════════════════════════════════════════════════════


class AttentionRolloutCapture:
    """Register hooks on TwoWayTransformer to capture attention weights,
    then compute attention rollout across layers.

    Architecture: depth=2 TwoWayAttentionBlock layers + final_attn
    Each block: self_attn → cross_attn_token_to_image → mlp → cross_attn_image_to_token

    Captured attention types (7 total per forward):
      layer[0]: self_attn             [B, 8, 6, 6]
      layer[0]: cross_attn_t2i        [B, 8, 6, 4096]
      layer[0]: cross_attn_i2t        [B, 8, 4096, 6]
      layer[1]: self_attn             [B, 8, 6, 6]
      layer[1]: cross_attn_t2i        [B, 8, 6, 4096]
      layer[1]: cross_attn_i2t        [B, 8, 4096, 6]
      final:    final_attn_t2i        [B, 8, 6, 4096]
    """

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.captured = {}  # layer_name → attention_weights
        self._register()

    def _register(self):
        """Register forward hooks on all Attention modules."""
        md = self.model.sam_decoder.mask_decoder
        tr = md.transformer

        for layer_idx in range(tr.depth):
            layer = tr.layers[layer_idx]

            def _make_hook(name):
                def _hook(module, input, output):
                    # Input: q, k, v tensors
                    # We need to recompute attention to capture attn weights.
                    # Simpler: patch the forward to return attn.
                    pass
                return _hook

            # Hook self_attn
            h1 = layer.self_attn.register_forward_hook(
                self._attention_hook(f"L{layer_idx}_self_attn")
            )
            self.hooks.append(h1)

            # Hook cross_attn_token_to_image
            h2 = layer.cross_attn_token_to_image.register_forward_hook(
                self._attention_hook(f"L{layer_idx}_cross_t2i")
            )
            self.hooks.append(h2)

            # Hook cross_attn_image_to_token
            h3 = layer.cross_attn_image_to_token.register_forward_hook(
                self._attention_hook(f"L{layer_idx}_cross_i2t")
            )
            self.hooks.append(h3)

        # Hook final_attn_token_to_image
        hf = tr.final_attn_token_to_image.register_forward_hook(
            self._attention_hook("final_cross_t2i")
        )
        self.hooks.append(hf)

    def _attention_hook(self, name):
        """Create a hook that recomputes attention weights from the attention module.

        We intercept the module's forward by patching it to also save attn weights.
        Since PyTorch hooks can't easily capture intermediate values, we do
        a trick: temporarily patch the module's forward method.
        """
        def _hook(module, args, output):
            # Recompute attention manually using stored inputs
            # args = (q, k, v) tensors from the module.forward() call
            # BUT: forward hook receives (module, input_args, output)
            # We can't easily get attn from output (it's just the projection)
            # WORKAROUND: use module internals to recalc
            pass
        # This approach is complex. Let's use a different strategy:
        # Directly monkey-patch the Attention.forward to return attn as well.
        return _hook

    def capture_attention_weights(self):
        """Monkey-patch all Attention.forward to also return attention weights.

        Call BEFORE running the decoder forward.
        """
        capture_self = self  # for closure access inside patched_forward

        md = self.model.sam_decoder.mask_decoder
        tr = md.transformer

        self._original_forwards = {}
        self._attn_weights = {}

        def _make_patched_forward(attn_module, name):
            # Capture module attributes once (faster than attribute lookup each call)
            q_proj = attn_module.q_proj
            k_proj = attn_module.k_proj
            v_proj = attn_module.v_proj
            out_proj = attn_module.out_proj
            num_heads = attn_module.num_heads
            internal_dim = attn_module.internal_dim

            def patched_forward(self_, q, k, v):
                qp = q_proj(q)
                kp = k_proj(k)
                vp = v_proj(v)

                B, Nq, _ = qp.shape
                _, Nk, _ = kp.shape
                qp = qp.reshape(B, Nq, num_heads, internal_dim // num_heads).transpose(1, 2)
                kp = kp.reshape(B, Nk, num_heads, internal_dim // num_heads).transpose(1, 2)
                vp = vp.reshape(B, Nk, num_heads, internal_dim // num_heads).transpose(1, 2)

                import math
                attn = qp @ kp.transpose(-2, -1) / math.sqrt(internal_dim // num_heads)
                attn_w = torch.softmax(attn, dim=-1)

                # Save attention weights
                capture_self._attn_weights[name] = attn_w.detach().cpu()

                out = attn_w @ vp
                out = out.transpose(1, 2).reshape(B, Nq, internal_dim)
                out = out_proj(out)
                return out
            return patched_forward

        # Patch all attention modules
        attn_modules = []
        for i in range(tr.depth):
            layer = tr.layers[i]
            attn_modules.extend([
                (layer.self_attn, f"L{i}_self"),
                (layer.cross_attn_token_to_image, f"L{i}_t2i"),
                (layer.cross_attn_image_to_token, f"L{i}_i2t"),
            ])
        attn_modules.append((tr.final_attn_token_to_image, "final_t2i"))

        for module, name in attn_modules:
            self._original_forwards[name] = module.forward
            patched = _make_patched_forward(module, name)
            module.forward = patched.__get__(module, type(module))

    def restore(self):
        """Restore original forward methods."""
        md = self.model.sam_decoder.mask_decoder
        tr = md.transformer
        attn_modules = []
        for i in range(tr.depth):
            layer = tr.layers[i]
            attn_modules.extend([
                (layer.self_attn, f"L{i}_self"),
                (layer.cross_attn_token_to_image, f"L{i}_t2i"),
                (layer.cross_attn_image_to_token, f"L{i}_i2t"),
            ])
        attn_modules.append((tr.final_attn_token_to_image, "final_t2i"))
        for module, name in attn_modules:
            if name in self._original_forwards:
                module.forward = self._original_forwards[name]
        self._original_forwards.clear()
        self._attn_weights.clear()

    def get_rollout(self) -> dict:
        """Compute attention rollout from captured weights.

        Rollout for token-to-image cross attention:
          rollout^{(L)} = A^{(L)} @ rollout^{(L-1)}  (each normalized)

        Returns:
          - per_layer_t2i_rollout: list of [4096] spatial maps
          - final_rollout: [4096] spatial attention map
          - token_self_attn_rollout: [6, 6] token interaction matrix
        """
        result = {"captured": dict(self._attn_weights)}

        # Token-to-image rollout: accumulate cross-attention across layers
        t2i_keys = [k for k in self._attn_weights if "t2i" in k]
        t2i_keys.sort()

        if t2i_keys:
            # For each t2i layer, average over heads and tokens → [N_image] map
            spatial_maps = []
            for key in t2i_keys:
                attn = self._attn_weights[key]  # [1, 8, 6 (or N_tokens), 4096]
                # Average over heads and tokens → [4096]
                sp_map = attn[0].mean(dim=0).mean(dim=0).numpy()  # [4096]
                spatial_maps.append(sp_map)

            # Rollout: start from first layer, multiply through
            rollout = spatial_maps[0].copy()
            for i in range(1, len(spatial_maps)):
                # Element-wise product (each pixel's attention is modulated)
                rollout = rollout * spatial_maps[i]
                rollout = rollout / (rollout.sum() + 1e-10)  # normalize

            result["spatial_rollout_layers"] = [m.tolist() for m in spatial_maps]
            result["spatial_rollout_final"] = rollout.tolist()

            # Reshape to 64x64
            rollout_2d = rollout.reshape(64, 64)
            result["spatial_rollout_64x64"] = rollout_2d.tolist()

            # Quantify: spatial concentration (Gini of rollout)
            sorted_vals = np.sort(rollout)
            n = len(sorted_vals)
            gini = 1 - 2 * np.sum(sorted_vals * np.arange(1, n + 1)) / (n * sorted_vals.sum())
            result["rollout_gini"] = float(gini)

            # Mean per-layer spatial entropy
            entropies = []
            for m in spatial_maps:
                p = m / m.sum()
                entropies.append(float(-np.sum(p * np.log(p + 1e-10)) / np.log(len(p))))
            result["spatial_entropy_per_layer"] = entropies

        # Token self-attention: token interaction matrix
        self_keys = [k for k in self._attn_weights if "_self" in k]
        self_keys.sort()
        if self_keys:
            # Average over heads and layers
            all_self = []
            for key in self_keys:
                attn = self._attn_weights[key]  # [1, 8, 6, 6]
                all_self.append(attn[0].mean(dim=0).numpy())  # [6, 6]
            token_mat = np.mean(all_self, axis=0)  # [6, 6]
            result["token_self_attn"] = token_mat.tolist()
            # Token roles: 0=iou, 1-4=mask_tokens, 5=sparse_prompt
            result["token_labels"] = ["iou", "mask0", "mask1", "mask2", "mask3", "sparse"]

        return result


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Attention Rollout — Where does the TwoWayTransformer look?"
    )
    add_common_args(parser)
    args = parser.parse_args()

    # Attention rollout only works with SAM decoder (not BypassHead)
    ctx = build_diag_context(args, require_adapter=True, split="val")
    if ctx.model.bypass_head is not None:
        print("ERROR: Attention Rollout requires SAM Decoder, not BypassMaskHead.")
        print("  Use a checkpoint trained without bypass_decoder=True.")
        print("  Or re-train with bypass_decoder=False in the YAML config.")
        sys.exit(1)

    build_support_cache(ctx)
    select_tiles(ctx, num_tiles=args.num_tiles)

    # Setup attention capture
    cap = AttentionRolloutCapture(ctx.model)
    cap.capture_attention_weights()

    per_tile = []

    for tile_idx, present_classes in tqdm(ctx.selected_tiles, desc="attention rollout"):
        sample = ctx.dataset[tile_idx]
        H, W = sample["image"].shape[1], sample["image"].shape[2]

        xx, _ = preprocess_image(sample["image"])
        query_emb = ctx.backbone(xx.unsqueeze(0).to(ctx.device))["image_embedding"]
        if ctx.adapter is not None:
            query_emb = ctx.adapter(query_emb)

        main_cls = max(present_classes, key=lambda c:
                       ctx.dataset.get_class_mask(tile_idx, c).sum())
        gt = ctx.dataset.get_class_mask(tile_idx, main_cls).numpy().astype(bool)
        sup_data = ctx.support_cache.get(main_cls)
        if sup_data is None:
            continue
        sup_feat, sup_mask = sup_data

        # Manual forward: run through the decoder to trigger attention hooks
        with torch.no_grad():
            support_memory = ctx.model.support_encoder(sup_feat, sup_mask)
            dense_pe = ctx.model.sam_decoder.prompt_encoder.get_dense_pe()
            spg_out = ctx.model.spg(query_emb, support_memory, dense_pe)

            if ctx.model.prompt_fusion is not None and ctx.model.geometric_prior is not None:
                gp = ctx.model.geometric_prior(query_emb, support_memory)
                dense_prompt, sparse_token = ctx.model.prompt_fusion(gp, spg_out.semantic_prior)
            else:
                dense_prompt = spg_out.semantic_prior
                sparse_token = dense_prompt.mean(dim=(2, 3))

            # Decode via SAM decoder (triggers TwoWayTransformer → hooks)
            support_proto = ctx.model._compute_support_prototype(sup_feat, sup_mask)
            saved_cat = ctx.model.sam_decoder._category_enabled
            ctx.model.sam_decoder._category_enabled = False
            low_res, _ = ctx.model.sam_decoder(
                query_emb, sparse_token, dense_prompt,
                support_prototype=support_proto,
            )
            ctx.model.sam_decoder._category_enabled = saved_cat

        # Extract attention rollout
        rollout = cap.get_rollout()

        # Compare rollout with GT
        if "spatial_rollout_64x64" in rollout:
            rollout_2d = np.array(rollout["spatial_rollout_64x64"])
            gt_t = torch.from_numpy(gt.astype(np.float32))
            gt_64 = F.interpolate(
                gt_t.unsqueeze(0).unsqueeze(0), (64, 64), mode="area"
            )[0, 0].numpy()

            # IoU between rollout top-K and GT
            for percentile in [50, 70, 80, 90]:
                threshold = np.percentile(rollout_2d, percentile)
                rollout_mask = rollout_2d > threshold
                inter = (rollout_mask & gt_64.astype(bool)).sum()
                union = (rollout_mask | gt_64.astype(bool)).sum()
                rollout[f"rollout_vs_gt_iou_p{percentile}"] = float(inter / union) if union > 0 else 0.0

            # Correlation
            rollout_flat = rollout_2d.ravel()
            gt_flat = gt_64.ravel()
            s_r, s_g = rollout_flat.std(), gt_flat.std()
            if s_r > 1e-8 and s_g > 1e-8:
                rollout["rollout_gt_pearson_r"] = float(np.corrcoef(rollout_flat, gt_flat)[0, 1])
            else:
                rollout["rollout_gt_pearson_r"] = 0.0

        entry = {
            "tile_idx": tile_idx,
            "class_id": main_cls,
            "class_name": ISAID5I_CATEGORIES.get(main_cls, f"cls{main_cls}"),
        }
        # Add rollout metrics
        for key in ["rollout_gini", "rollout_gt_pearson_r",
                     "rollout_vs_gt_iou_p50", "rollout_vs_gt_iou_p70",
                     "rollout_vs_gt_iou_p80", "rollout_vs_gt_iou_p90"]:
            if key in rollout:
                entry[key] = rollout[key]
        if "spatial_entropy_per_layer" in rollout:
            for i, ent in enumerate(rollout["spatial_entropy_per_layer"]):
                entry[f"L{i}_spatial_entropy"] = ent

        per_tile.append(entry)

    # Restore patched attention modules
    cap.restore()

    # ── Aggregate ──
    summary = {"n_tiles": len(per_tile)}
    summary["metrics"] = aggregate_metrics(per_tile)

    s_path, p_path = save_results(ctx.out_dir, summary, per_tile)

    # ── Print report ──
    print_header("Attention Rollout — TwoWayTransformer Spatial Focus")
    if "metrics" in summary:
        print("\n  ┌─ Rollout vs GT Alignment")
        print_metric_table({k: v for k, v in summary["metrics"].items()
                           if "iou" in k or "pearson" in k})

        print("\n  ┌─ Rollout Statistics")
        print_metric_table({k: v for k, v in summary["metrics"].items()
                           if "gini" in k or "entropy" in k})

    print(f"\n[Summary] {s_path}")
    print(f"[Per-tile] {p_path}")
    print("[Done]")


if __name__ == "__main__":
    main()
