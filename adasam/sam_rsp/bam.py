"""
BAM (Base-model Adaptation Module) — RSPG (Rough Segment Prompt Generator).
============================================================================

PFENet-style few-shot segmentation meta-learner.  Uses a frozen PSPNet backbone
+ frozen base learner (PPM + cls) from Stage 1, and trains a meta learner on top.

Architecture::

    Query image ──→ [Frozen Backbone] ──→ query_feat_3, query_feat_4
    Support imgs ──→ [Frozen Backbone] ──→ supp_feat_3, supp_feat_4 (masked)

    supp_feat_3 + mask ──→ Weighted GAP ──→ supp_prototype [B, 256, 1, 1]
    query_feat_4, supp_feat_4 ──→ Cosine Similarity ──→ prior_mask

    [query_feat, supp_proto, prior_mask] ──→ init_merge ──→ ASPP ──→ res blocks
        ──→ cls_meta ──→ meta_out (FG/BG)

    query_feat_4 ──→ [Frozen Base Learner] ──→ base_out (multi-class)

    meta_out + base_out ──→ Ensemble ──→ final_out (FG/BG)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# ASPP (Atrous Spatial Pyramid Pooling)
# ═══════════════════════════════════════════════════════════════════

class ASPP(nn.Module):
    def __init__(self, in_channels=3, out_channels=256):
        super().__init__()
        self.layer6_0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.ReLU(),
        )
        self.layer6_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.ReLU(),
        )
        self.layer6_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=True),
            nn.ReLU(),
        )
        self.layer6_3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=True),
            nn.ReLU(),
        )
        self.layer6_4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=True),
            nn.ReLU(),
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        feature_size = x.shape[-2:]
        global_feature = F.avg_pool2d(x, kernel_size=feature_size)
        global_feature = self.layer6_0(global_feature)
        global_feature = global_feature.expand(-1, -1, feature_size[0], feature_size[1])
        return torch.cat(
            [global_feature, self.layer6_1(x), self.layer6_2(x), self.layer6_3(x), self.layer6_4(x)], dim=1
        )


# ═══════════════════════════════════════════════════════════════════
# PPM (Pyramid Pooling Module)
# ═══════════════════════════════════════════════════════════════════

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super().__init__()
        self.features = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(b),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True),
            ) for b in bins
        ])

    def forward(self, x):
        x_size = x.size()
        return torch.cat([x] + [
            F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True)
            for f in self.features
        ], dim=1)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def weighted_gap(supp_feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Weighted Global Average Pooling: mask-guided prototype extraction.

    Args:
        supp_feat: [B, C, H, W]
        mask:      [B, 1, H, W]
    Returns:
        prototype: [B, C, 1, 1]
    """
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:]
    area = F.avg_pool2d(mask, (feat_h, feat_w)) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(supp_feat, (feat_h, feat_w)) * feat_h * feat_w / area
    return supp_feat


def get_gram_matrix(fea: torch.Tensor) -> torch.Tensor:
    """Compute gram matrix for feature quality estimation.

    Args:
        fea: [B, C, H, W]
    Returns:
        gram: [B, C, C] in (0, 1)
    """
    b, c, h, w = fea.shape
    fea = fea.reshape(b, c, h * w)
    fea_T = fea.permute(0, 2, 1)
    fea_norm = fea.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T) / (torch.bmm(fea_norm, fea_T_norm) + 1e-7)
    return gram


# ═══════════════════════════════════════════════════════════════════
# BAM Model
# ═══════════════════════════════════════════════════════════════════

class BAMModel(nn.Module):
    """BAM (Base-model Adaptation Module / RSPG).

    PFENet-style few-shot segmentation meta-learner.
    """

    def __init__(
        self,
        num_base_classes: int,
        shot: int = 1,
        layers: int = 50,
        vgg: bool = False,
        low_fea: str = 'layer2',
        kshot_trans_dim: int = 2,
        zoom_factor: int = 8,
    ):
        super().__init__()
        self.shot = shot
        self.layers = layers
        self.vgg = vgg
        self.low_fea = low_fea
        self.zoom_factor = zoom_factor
        self.num_classes = 2  # FG/BG
        self.base_classes = num_base_classes

        # ── Build PSPNet backbone from torchvision ResNet ──
        self._build_backbone()
        self.fea_dim = 512 if vgg else 2048

        # ── Base Learner (PPM + classifier) ──
        bins = (1, 2, 3, 6)
        ppm_reduction = int(self.fea_dim / len(bins))
        self.ppm = PPM(self.fea_dim, ppm_reduction, bins)
        self.base_cls = nn.Sequential(
            nn.Conv2d(self.fea_dim * 2, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(512, num_base_classes + 1, kernel_size=1),
        )

        # ── Meta Learner ──
        reduce_dim = 256
        # Mid-level feature dimension: layer2(512) + layer3(1024) for ResNet50
        mid_feat_dim = 512 + 256 if vgg else 1024 + 512
        self.down_query = nn.Sequential(
            nn.Conv2d(mid_feat_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(mid_feat_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
        )
        mask_add_num = 1
        self.init_merge = nn.Sequential(
            nn.Conv2d(reduce_dim * 2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.ASPP_meta = ASPP(reduce_dim)
        self.res1_meta = nn.Sequential(
            nn.Conv2d(reduce_dim * 5, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.res2_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.cls_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, self.num_classes, kernel_size=1),
        )

        # Gram and Meta merge
        self.gram_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.gram_merge.weight = nn.Parameter(
            torch.tensor([[1.0], [0.0]]).reshape_as(self.gram_merge.weight)
        )

        # Learner Ensemble (base + meta)
        self.cls_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.cls_merge.weight = nn.Parameter(
            torch.tensor([[1.0], [0.0]]).reshape_as(self.cls_merge.weight)
        )

        # K-Shot Reweighting
        if shot > 1:
            if kshot_trans_dim == 0:
                self.kshot_rw = nn.Conv2d(shot, shot, kernel_size=1, bias=False)
                self.kshot_rw.weight = nn.Parameter(
                    torch.ones_like(self.kshot_rw.weight) / shot
                )
            else:
                self.kshot_rw = nn.Sequential(
                    nn.Conv2d(shot, kshot_trans_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(kshot_trans_dim, shot, kernel_size=1),
                )

        self._initialized = False

    def _build_backbone(self):
        """Build ResNet50 backbone with dilated convolutions (same as Stage 1)."""
        try:
            from torchvision.models import resnet50, ResNet50_Weights
        except ImportError:
            from torchvision.models import resnet50
            ResNet50_Weights = None

        if ResNet50_Weights is not None:
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            resnet = resnet50(weights=None)

        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Apply dilation
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

    def load_stage1_weights(self, checkpoint_path: str) -> None:
        """Load Stage 1 PSPNet weights into backbone + base learner."""
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))

        # Stage 1 saves: encoder.0.* = layer0, encoder.1.* = layer1, etc.
        # Remap: encoder.{idx}.* → layer{idx}.*
        layer_idx_map = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4'}
        our_state = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.'):
                parts = k.split('.', 2)  # ['encoder', '{idx}', 'rest...']
                if parts[1] in layer_idx_map:
                    new_k = f'layer{parts[1]}.{parts[2]}'
                    our_state[new_k] = v
            elif k.startswith('ppm.'):
                our_state[k] = v
            elif k.startswith('cls.'):
                our_state[f'base_cls.{k[4:]}'] = v

        missing, unexpected = self.load_state_dict(our_state, strict=False)
        n_loaded = len(our_state)
        print(f"[BAM] Loaded {n_loaded} params from {checkpoint_path}")
        if missing:
            # Only report non-meta-learner missing keys (meta learner is newly initialized)
            meta_keys = [k for k in missing if not any(
                p in k for p in ['down_query', 'down_supp', 'init_merge', 'ASPP_meta',
                                 'res1_meta', 'res2_meta', 'cls_meta', 'gram_merge',
                                 'cls_merge', 'kshot_rw']
            )]
            if meta_keys:
                print(f"[BAM] Missing backbone keys ({len(meta_keys)}): {meta_keys[:5]}...")
        self._initialized = True

    def freeze_backbone_and_base(self):
        """Freeze backbone (layers 0-4) and base learner (ppm + cls)."""
        for param in self.layer0.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
        for param in self.layer3.parameters():
            param.requires_grad = False
        for param in self.layer4.parameters():
            param.requires_grad = False
        for param in self.ppm.parameters():
            param.requires_grad = False
        for param in self.base_cls.parameters():
            param.requires_grad = False

    def _extract_query_features(self, x: torch.Tensor):
        """Extract query features through frozen backbone."""
        with torch.no_grad():
            q0 = self.layer0(x)
            q1 = self.layer1(q0)
            q2 = self.layer2(q1)
            q3 = self.layer3(q2)
            q4 = self.layer4(q3)
        return q0, q1, q2, q3, q4

    def _extract_support_features(self, s_x_i: torch.Tensor, mask_i: torch.Tensor):
        """Extract support features for one shot through frozen backbone."""
        with torch.no_grad():
            s0 = self.layer0(s_x_i)
            s1 = self.layer1(s0)
            s2 = self.layer2(s1)
            s3 = self.layer3(s2)
            mask_resized = F.interpolate(
                mask_i, size=(s3.size(2), s3.size(3)),
                mode='bilinear', align_corners=True,
            )
            s4 = self.layer4(s3 * mask_resized)
        return s0, s1, s2, s3, s4, mask_resized

    def forward(
        self,
        query_img: torch.Tensor,       # [B, 3, H, W]
        support_imgs: torch.Tensor,    # [B, K, 3, H, W]
        support_masks: torch.Tensor,   # [B, K, H, W]  binary (1=FG)
        cat_idx: torch.Tensor | None = None,  # [B] or [1, B]
    ):
        x_size = query_img.size()
        bs = x_size[0]
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        # ── Query features ──
        _, _, q2, q3, q4 = self._extract_query_features(query_img)

        if self.vgg:
            q2_up = F.interpolate(q2, size=(q3.size(2), q3.size(3)),
                                  mode='bilinear', align_corners=True)
            query_feat = torch.cat([q3, q2_up], 1)
        else:
            query_feat = torch.cat([q3, q2], 1)
        query_feat = self.down_query(query_feat)

        # ── Support features (per shot) ──
        supp_pro_list = []
        final_supp_list = []
        mask_list = []
        supp_feat_list = []  # for gram matrix

        for i in range(self.shot):
            mask_i = (support_masks[:, i, :, :] == 1).float().unsqueeze(1)  # [B, 1, H, W]
            mask_list.append(mask_i)

            _, _, s2, s3, s4, mask_resized = self._extract_support_features(
                support_imgs[:, i, :, :, :], mask_i
            )
            final_supp_list.append(s4)

            if self.vgg:
                s2_up = F.interpolate(s2, size=(s3.size(2), s3.size(3)),
                                      mode='bilinear', align_corners=True)
                supp_feat_mid = torch.cat([s3, s2_up], 1)
            else:
                supp_feat_mid = torch.cat([s3, s2], 1)

            supp_feat_mid = self.down_supp(supp_feat_mid)
            supp_pro = weighted_gap(supp_feat_mid, mask_resized)
            supp_pro_list.append(supp_pro)

            # Store low-level feat for gram matrix
            supp_feat_list.append(eval(f's{self.low_fea[-1]}'))

        # ── K-Shot Reweighting ──
        que_gram = get_gram_matrix(eval(f'q{self.low_fea[-1]}'))
        norm_max = torch.ones_like(que_gram).norm(dim=(1, 2))
        est_val_list = []
        for supp_item in supp_feat_list:
            supp_gram = get_gram_matrix(supp_item)
            gram_diff = que_gram - supp_gram
            est_val_list.append(
                (gram_diff.norm(dim=(1, 2)) / norm_max).reshape(bs, 1, 1, 1)
            )
        est_val_total = torch.cat(est_val_list, 1)  # [B, K, 1, 1]
        if self.shot > 1:
            val1, idx1 = est_val_total.sort(1)
            val2, idx2 = idx1.sort(1)
            weight = self.kshot_rw(val1)
            weight = weight.gather(1, idx2)
            weight_soft = torch.softmax(weight, 1)
        else:
            weight_soft = torch.ones_like(est_val_total)
        est_val = (weight_soft * est_val_total).sum(1, True)  # [B, 1, 1, 1]

        # ── Prior Similarity Mask ──
        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(
                mask_list[i], size=(resize_size, resize_size),
                mode='bilinear', align_corners=True,
            )
            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask

            q = q4
            s = tmp_supp_feat_4
            bsize_, ch_sz, sp_sz, _ = q.size()

            tmp_query = q.reshape(bsize_, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s.reshape(bsize_, ch_sz, -1).permute(0, 2, 1)
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

            similarity = torch.bmm(tmp_supp, tmp_query) / (
                torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps
            )
            similarity = similarity.max(1)[0].reshape(bsize_, sp_sz * sp_sz)
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps
            )
            corr_query = similarity.reshape(bsize_, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(
                corr_query, size=(q3.size(2), q3.size(3)),
                mode='bilinear', align_corners=True,
            )
            corr_query_mask_list.append(corr_query)
        corr_query_mask = torch.cat(corr_query_mask_list, 1)
        corr_query_mask = (weight_soft * corr_query_mask).sum(1, True)

        # ── Support Prototype ──
        supp_pro = torch.cat(supp_pro_list, 2)  # [B, 256, K, 1]
        supp_pro = (weight_soft.permute(0, 2, 1, 3) * supp_pro).sum(2, True)

        # ── Merge ──
        concat_feat = supp_pro.expand_as(query_feat)
        merge_feat = torch.cat([query_feat, concat_feat, corr_query_mask], 1)
        merge_feat = self.init_merge(merge_feat)

        # ── Base and Meta ──
        base_out = self.ppm(q4)
        base_out = self.base_cls(base_out)

        query_meta = self.ASPP_meta(merge_feat)
        query_meta = self.res1_meta(query_meta)
        query_meta = self.res2_meta(query_meta) + query_meta
        meta_out = self.cls_meta(query_meta)

        meta_out_soft = meta_out.softmax(1)
        base_out_soft = base_out.softmax(1)

        # ── Classifier Ensemble ──
        meta_map_bg = meta_out_soft[:, 0:1, :, :]
        meta_map_fg = meta_out_soft[:, 1:, :, :]

        if self.training and cat_idx is not None:
            # During training on base classes: merge all non-target base classes as FG
            c_id_array = torch.arange(self.base_classes + 1, device=query_img.device)
            base_map_list = []
            for b_id in range(bs):
                if cat_idx.dim() == 2:
                    c_id = cat_idx[0][b_id].item() + 1
                else:
                    c_id = cat_idx[b_id].item() + 1
                c_mask = (c_id_array != 0) & (c_id_array != c_id)
                base_map_list.append(
                    base_out_soft[b_id, c_mask, :, :].unsqueeze(0).sum(1, True)
                )
            base_map = torch.cat(base_map_list, 0)
        else:
            # Inference on novel classes: sum all non-BG as FG
            base_map = base_out_soft[:, 1:, :, :].sum(1, True)

        est_map = est_val.expand_as(meta_map_fg)

        meta_map_bg = self.gram_merge(torch.cat([meta_map_bg, est_map], dim=1))
        meta_map_fg = self.gram_merge(torch.cat([meta_map_fg, est_map], dim=1))

        merge_map = torch.cat([meta_map_bg, base_map], 1)
        merge_bg = self.cls_merge(merge_map)

        final_out = torch.cat([merge_bg, meta_map_fg], dim=1)

        # ── Upsample ──
        if self.zoom_factor != 1:
            final_out = F.interpolate(
                final_out, size=(h, w), mode='bilinear', align_corners=True,
            )
            base_out = F.interpolate(
                base_out, size=(h, w), mode='bilinear', align_corners=True,
            )

        return final_out, base_out

    def get_trainable_params(self):
        """Return trainable parameter groups for optimizer."""
        if self.shot > 1:
            param_groups = [
                {'params': self.down_query.parameters()},
                {'params': self.down_supp.parameters()},
                {'params': self.init_merge.parameters()},
                {'params': self.ASPP_meta.parameters()},
                {'params': self.res1_meta.parameters()},
                {'params': self.res2_meta.parameters()},
                {'params': self.cls_meta.parameters()},
                {'params': self.gram_merge.parameters()},
                {'params': self.cls_merge.parameters()},
                {'params': self.kshot_rw.parameters()},
            ]
        else:
            param_groups = [
                {'params': self.down_query.parameters()},
                {'params': self.down_supp.parameters()},
                {'params': self.init_merge.parameters()},
                {'params': self.ASPP_meta.parameters()},
                {'params': self.res1_meta.parameters()},
                {'params': self.res2_meta.parameters()},
                {'params': self.cls_meta.parameters()},
                {'params': self.gram_merge.parameters()},
                {'params': self.cls_merge.parameters()},
            ]
        return param_groups
