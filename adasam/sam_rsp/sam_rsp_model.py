"""
SAM-RSP Full Model — Stage 3.
==============================

Complete SAM-RSP architecture for few-shot segmentation::

    Query ──→ [Frozen BAM] ──→ rough_mask ──────────────┐
    Query ──→ [Frozen SAM ViT-H] ──→ query_feat ────────┤
    Support ──→ [Frozen BAM] ──→ ...                    │
                                                         │
    query_feat · rough_mask ──→ pixel_prototype ────────┤
                                                         │
    [query_feat ⊕ prototype ⊕ rough_mask ⊕ diff_mask] ──┤
        ──→ [Trainable ViT Blocks × N] ──→ FG/BG output

Key dimensions (1024² input):
    query_feat_sam:  [B, 256, 64, 64]
    pixel_prototype: [B, 256, 64, 64]
    rough_mask:      [B, 2, 64, 64]
    diff_mask:       [B, 1, 64, 64]
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAM_RSP = _REPO_ROOT / "thirdparty" / "SAM-RSP"
if not _SAM_RSP.is_dir():
    raise ImportError(
        f"SAM-RSP thirdparty directory not found: {_SAM_RSP}\n"
        "Clone it first:\n"
        f"  cd {_REPO_ROOT / 'thirdparty'}\n"
        "  git clone https://github.com/nironbow/SAM-RSP.git\n"
        "Or download and extract from https://github.com/nironbow/SAM-RSP"
    )
if str(_SAM_RSP) not in sys.path:
    sys.path.insert(0, str(_SAM_RSP))

from model.image_encoder import ImageEncoderViT, Block
from model.common import MLPBlock, LayerNorm2d


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def create_diff_mask(output_bin: torch.Tensor) -> torch.Tensor:
    """Create difference mask: FG score - BG score, normalized to [0,1].

    Args:
        output_bin: [B, 2, H, W] — FG/BG logits
    Returns:
        diff_mask: [B, 1, H, W]
    """
    cosine_eps = 1e-7
    bsize, ch, h, w = output_bin.size()
    diff = (output_bin[:, 1, :, :] - output_bin[:, 0, :, :]).reshape(bsize, -1)
    diff_max = torch.max(diff, dim=1, keepdim=True)[0]
    diff_min = torch.min(diff, dim=1, keepdim=True)[0]
    diff = (diff - diff_min) / (diff_max - diff_min + cosine_eps)
    return diff.reshape(bsize, 1, h, w)


# ═══════════════════════════════════════════════════════════════════
# SAM-RSP Full Model
# ═══════════════════════════════════════════════════════════════════

class SAMRSPModel(nn.Module):
    """SAM-RSP full model for Stage 3 training.

    Frozen: BAM (RSPG) + SAM ViT-H encoder
    Trainable: init_merge + ViT decoder blocks + cls heads
    """

    def __init__(
        self,
        bam_model: nn.Module,           # Pre-built BAM from Stage 2
        sam_checkpoint: str | None = None,
        decoder_depth: int = 3,
        reduce_dim: int = 256,
        num_classes: int = 2,
    ):
        super().__init__()
        self.decoder_depth = decoder_depth
        self.reduce_dim = reduce_dim
        self.num_classes = num_classes

        # ── BAM (RSPG) — frozen ──
        self.bam = bam_model
        for param in self.bam.parameters():
            param.requires_grad = False

        # ── SAM ViT-H Encoder — frozen ──
        self.sam_encoder = ImageEncoderViT(
            img_size=1024,
            patch_size=16,
            depth=24,
            num_heads=16,
            embed_dim=1024,
            use_abs_pos=True,
            use_rel_pos=True,
            rel_pos_zero_init=True,
            global_attn_indexes=[5, 11, 17, 23],
            window_size=14,
        )
        if sam_checkpoint is not None and Path(sam_checkpoint).exists():
            self._load_sam_weights(sam_checkpoint)
        for param in self.sam_encoder.parameters():
            param.requires_grad = False

        # ── Decoder: init_merge ──
        # Input: query_feat(256) + prototype(256) + bam_out(2) + diff_mask(1) = 515
        init_in_ch = reduce_dim * 2 + num_classes + 1  # 515
        self.init_merge = nn.Sequential(
            nn.Conv2d(init_in_ch, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        # ── Decoder: ViT blocks ──
        self.blocks = nn.ModuleList()
        for i in range(decoder_depth):
            block = Block(
                dim=reduce_dim,        # 256
                num_heads=16,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU,
                use_rel_pos=True,
                rel_pos_zero_init=True,
                window_size=14,
                input_size=(64, 64),
            )
            self.blocks.append(block)

        # ── Decoder: inner classifiers + beta convs ──
        self.inner_cls = nn.ModuleList()
        self.beta_conv = nn.ModuleList()
        for _ in range(decoder_depth - 1):
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, 8, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(8, num_classes, kernel_size=1),
            ))
            # beta_conv: 256 (merge_feat) + 2 (inner_out) + 1 (diff) = 259 → 256
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim + num_classes + 1, reduce_dim,
                          kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
            ))

        # ── Decoder: final head ──
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, num_classes, kernel_size=1),
        )

    def _load_sam_weights(self, checkpoint_path: str) -> None:
        """Load SAM ViT-H encoder weights."""
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        model_dict = self.sam_encoder.state_dict()
        matched = {}
        for k, v in ckpt.items():
            if 'image_encoder' in k:
                nk = k.replace('image_encoder.', '')
                if nk in model_dict:
                    matched[nk] = v
        self.sam_encoder.load_state_dict(matched, strict=False)
        print(f"[SAM-RSP] Loaded {len(matched)} SAM encoder params from {checkpoint_path}")

    def _get_pixel_prototype(
        self, query_feat_sam: torch.Tensor, bam_output: torch.Tensor
    ) -> torch.Tensor:
        """Compute pixel-level prototype via self-correlation + BAM guidance.

        Args:
            query_feat_sam: [B, 256, 64, 64]
            bam_output:     [B, 2, 64, 64]  (BAM rough FG/BG logits)
        Returns:
            prototype:      [B, 256, 64, 64]
        """
        bs, ch, h, w = query_feat_sam.size()
        cosine_eps = 1e-7

        # Self-correlation matrix
        tmp_query = query_feat_sam.reshape(bs, ch, -1)           # [B, 256, 4096]
        query_extract = tmp_query.permute(0, 2, 1)                # [B, 4096, 256]
        corr = torch.bmm(query_extract, tmp_query) / math.sqrt(ch)  # [B, 4096, 4096]

        # Normalize
        min_vals, _ = torch.min(corr, dim=2, keepdim=True)
        max_vals, _ = torch.max(corr, dim=2, keepdim=True)
        corr = (corr - min_vals) / (max_vals - min_vals + cosine_eps)

        # BAM guidance: FG-BG difference, clamp negative to 0
        diff = (bam_output[:, 1:2, :, :] - bam_output[:, 0:1, :, :]).reshape(bs, 1, -1)
        diff = torch.clamp(diff, min=0)

        corr = corr * diff
        corr = F.threshold(corr, 0, -1e7)
        corr = F.softmax(corr, dim=-1)

        # Weighted sum
        query_clone = query_feat_sam.reshape(bs, ch, -1, 1)  # [B, 256, 4096, 1]
        proto_list = []
        for i in range(bs):
            proto_i = corr[i] @ query_clone[i]  # [4096, 4096] @ [256, 4096, 1]
            proto_list.append(proto_i)
        prototype = torch.cat(proto_list, dim=0)              # [B*256, 4096, 1]
        prototype = prototype.reshape(bs, ch, h, w)           # [B, 256, 64, 64]
        return prototype

    def forward(
        self,
        query_img: torch.Tensor,       # [B, 3, H, W]
        support_imgs: torch.Tensor,    # [B, K, 3, H, W]
        support_masks: torch.Tensor,   # [B, K, H, W]
        cat_idx: torch.Tensor | None = None,  # [B]
    ):
        bs, _, h, w = query_img.size()

        # ── Step 1: BAM rough segment prompt ──
        with torch.no_grad():
            bam_out, _ = self.bam(query_img, support_imgs, support_masks, cat_idx)
            # bam_out: [B, 2, H', W'] — same spatial as query after BAM's zoom_factor

            # Resize BAM output to 64×64 for SAM decoder
            bam_out_64 = F.interpolate(
                bam_out, size=(64, 64), mode='bilinear', align_corners=True,
            )
            diff_mask = create_diff_mask(bam_out_64)  # [B, 1, 64, 64]

            # ── Step 2: SAM ViT encoder ──
            if h != 1024 or w != 1024:
                q_sam = F.interpolate(
                    query_img, size=(1024, 1024), mode='bilinear', align_corners=True,
                )
            else:
                q_sam = query_img
            query_feat_sam = self.sam_encoder(q_sam)  # [B, 256, 64, 64]

        # ── Step 3: Pixel prototype ──
        pixel_proto = self._get_pixel_prototype(query_feat_sam, bam_out_64)

        # ── Step 4: Merge ──
        merge_feat = torch.cat(
            [query_feat_sam, pixel_proto, bam_out_64, diff_mask], dim=1,
        )  # [B, 515, 64, 64]
        merge_feat = self.init_merge(merge_feat)          # [B, 256, 64, 64]
        merge_feat = merge_feat.permute(0, 2, 3, 1)      # [B, 64, 64, 256] (NHWC)

        # ── Step 5: ViT decoder blocks with deep supervision ──
        aux_outputs = []
        for idx, blk in enumerate(self.blocks):
            blk_input = copy.deepcopy(merge_feat)
            merge_feat = blk(blk_input)                   # [B, 64, 64, 256] (NHWC)

            if idx < len(self.inner_cls):
                # Inner classifier on the block INPUT (before transformation)
                feats_nchw = blk_input.permute(0, 3, 1, 2)        # [B, 256, 64, 64]
                inner_out = self.inner_cls[idx](feats_nchw)       # [B, 2, 64, 64]
                inner_diff = create_diff_mask(inner_out)           # [B, 1, 64, 64]
                aux_outputs.append(inner_out)

                # Inject inner output back: cat in NHWC → permute → beta_conv → permute back
                merge_feat_c = torch.cat(
                    [merge_feat,
                     inner_out.permute(0, 2, 3, 1),
                     inner_diff.permute(0, 2, 3, 1)],
                    dim=-1,
                )  # [B, 64, 64, 259]
                merge_feat_c = merge_feat_c.permute(0, 3, 1, 2)   # [B, 259, 64, 64]
                merge_feat_c = self.beta_conv[idx](merge_feat_c)  # [B, 256, 64, 64]
                merge_feat = merge_feat_c.permute(0, 2, 3, 1)     # [B, 64, 64, 256]

        # ── Step 6: Final head ──
        merge_feat = merge_feat.permute(0, 3, 1, 2)  # [B, 256, 64, 64]
        query_feat = self.res2(merge_feat) + merge_feat
        final_out = self.cls_head(query_feat)          # [B, 2, 64, 64]

        # ── Step 7: Upsample to output size ──
        final_out = F.interpolate(
            final_out, size=(h, w), mode='bilinear', align_corners=True,
        )

        return final_out, aux_outputs, bam_out

    def get_trainable_params(self):
        """Return trainable parameter groups for optimizer."""
        return [
            {'params': self.init_merge.parameters()},
            {'params': self.inner_cls.parameters()},
            {'params': self.beta_conv.parameters()},
            {'params': self.blocks.parameters()},
            {'params': self.res2.parameters()},
            {'params': self.cls_head.parameters()},
        ]
