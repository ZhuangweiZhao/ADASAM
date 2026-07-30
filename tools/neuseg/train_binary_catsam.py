"""
NEU-SEG Binary CAT-SAM Training | 二值化 CAT-SAM 训练.
=====================================================

直接用 CATAdapter + SAM Decoder 做二值缺陷分割, 测试管道上限.
Binary defect segmentation with CATAdapter + SAM Decoder to
measure the pipeline upper bound.

架构 | Architecture::

    Image [3, 200, 200] → pad to 1024²
    → MobileSAM Encoder (frozen) → [256, 64, 64]
    → CATAdapter (trainable, ~70K) → [256, 64, 64]
    → BinaryPromptGenerator (trainable)
        ├→ sparse_token [256]
        └→ dense_prompt [256, 64, 64]
    → SAM MaskDecoder → mask [1, 256, 256]
    → upsample to 200² → BCE + Focal + Dice Loss

用法 | Usage::

    # 仅训练 adapter + prompt_generator (decoder 冻结 — 测纯适配上限)
    python tools/neuseg/train_binary_catsam.py --epochs 200

    # 同时训练 decoder (放开 SAM 上限)
    python tools/neuseg/train_binary_catsam.py --epochs 200 --train-decoder

    # 冒烟测试
    python tools/neuseg/train_binary_catsam.py --epochs 1 --steps 5

    # 冻结 adapter (测 decoder-only 上限)
    python tools/neuseg/train_binary_catsam.py --epochs 200 --train-decoder --freeze-adapter
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adasam.adapters import CATAdapter
from adasam.backbone import MobileSAMBackbone, build_mobile_sam
from adasam.datasets import NEUSegDataset
from adasam.decoder.sam_mask_decoder import SemanticMaskDecoder, SemanticMaskDecoderConfig
from adasam.utils import set_seed
from adasam.utils.transforms import preprocess_image


# ═══════════════════════════════════════════════════════════════════════
# BinaryPromptGenerator
# ═══════════════════════════════════════════════════════════════════════

class BinaryPromptGenerator(nn.Module):
    """二值分割 Prompt 生成器 | Binary segmentation prompt generator.

    从适配后的特征生成 SAM Decoder 所需的 sparse_token + dense_prompt.
    Generates sparse_token + dense_prompt from adapted features for SAM Decoder.

    设计 | Design:
        - sparse_token: 全局上下文 → MLP → 单 token (图像条件化).
          Global context → MLP → single token (image-conditioned).
        - dense_prompt: 1×1 Conv 空间投影 (逐位置条件化).
          1×1 Conv spatial projection (per-location conditioning).
        - 末层零初始化 → 训练起点等价于恒等映射 (与 CATAdapter 精神一致).
          Zero-init final layers → training starts as identity (same spirit as CATAdapter).

    :param dim: 特征维度 (256 for MobileSAM).
    :param bottleneck: MLP 瓶颈维度.
    """

    def __init__(self, dim: int = 256, bottleneck: int = 64) -> None:
        super().__init__()
        self.dim = dim
        self.bottleneck = bottleneck

        # Dense prompt: 1×1 conv spatial projection
        self.dense_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        # Sparse token: global pooling → MLP
        self.sparse_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, bottleneck, bias=True),
            nn.GELU(),
            nn.Linear(bottleneck, dim, bias=True),
        )

        # ── Zero-init final layers (training starts as identity) ──
        nn.init.zeros_(self.dense_proj.weight)
        nn.init.zeros_(self.sparse_head[-1].weight)
        nn.init.zeros_(self.sparse_head[-1].bias)

        self._n_params = sum(p.numel() for p in self.parameters())

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        :param features: [B, C, H, W] adapted features.
        :return: (sparse_token [B, C], dense_prompt [B, C, H, W]).
        """
        dense = self.dense_proj(features)
        sparse = self.sparse_head(features)
        return sparse, dense

    def extra_repr(self) -> str:
        return f"dim={self.dim}, bottleneck={self.bottleneck}, params={self._n_params:,}"


# ═══════════════════════════════════════════════════════════════════════
# Loss Functions
# ═══════════════════════════════════════════════════════════════════════

def binary_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 3.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Binary focal loss with logits input.

    :param logits: [B, H, W] raw logits (before sigmoid).
    :param target: [B, H, W] binary {0, 1}.
    :param gamma: focal exponent (lower for binary since FG/BG less extreme than multi-class).
    :param eps: numerical stability.
    :return: scalar loss.
    """
    prob = torch.sigmoid(logits)
    prob = torch.clamp(prob, eps, 1.0 - eps)
    ce = F.binary_cross_entropy(prob, target, reduction="none")
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    focal_weight = (1.0 - p_t) ** gamma
    return (focal_weight * ce).mean()


def binary_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Binary soft Dice loss.

    :param logits: [B, H, W] raw logits.
    :param target: [B, H, W] binary {0, 1}.
    :param smooth: Laplace smoothing.
    :return: scalar loss ∈ [0, 1].
    """
    prob = torch.sigmoid(logits)
    # Flatten spatial dims per sample
    prob_flat = prob.reshape(prob.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    intersection = (prob_flat * target_flat).sum(dim=1)
    union = prob_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return (1.0 - dice).mean()


def combined_binary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 0.3,
    focal_weight: float = 0.3,
    dice_weight: float = 0.4,
    focal_gamma: float = 3.0,
    focal_eps: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined binary segmentation loss: BCE + Focal + Dice.

    :return: (total_loss, {"bce": float, "focal": float, "dice": float}).
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
    focal = binary_focal_loss(logits, target, gamma=focal_gamma, eps=focal_eps)
    dice = binary_dice_loss(logits, target)
    total = bce_weight * bce + focal_weight * focal + dice_weight * dice
    return total, {"bce": bce.item(), "focal": focal.item(), "dice": dice.item()}


# ═══════════════════════════════════════════════════════════════════════
# Data Augmentation
# ═══════════════════════════════════════════════════════════════════════

class SegAug:
    """轻量数据增强 | Lightweight data augmentation (same as neuseg/train.py)."""

    def __init__(
        self,
        p_flip: float = 0.5,
        p_rotate: float = 0.5,
        brightness: float = 0.2,
        contrast: float = 0.2,
        noise_std: float = 0.02,
    ):
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std

    def __call__(
        self, img: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply random augmentation.

        :param img: [3, H, W] float ∈ [0, 1].
        :param mask: [1, H, W] binary {0, 1}.
        :return: augmented (img, mask).
        """
        if random.random() < self.p_flip:
            img = torch.flip(img, [-1])
            mask = torch.flip(mask, [-1])
        if random.random() < self.p_flip:
            img = torch.flip(img, [-2])
            mask = torch.flip(mask, [-2])
        if random.random() < self.p_rotate:
            k = random.randint(0, 3)
            img = torch.rot90(img, k, [-2, -1])
            mask = torch.rot90(mask, k, [-2, -1])
        b = random.uniform(-self.brightness, self.brightness)
        c = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
        img = torch.clamp(img * c + b, 0.0, 1.0)
        if self.noise_std > 0:
            img = torch.clamp(
                img + torch.randn_like(img) * self.noise_std, 0.0, 1.0
            )
        return img, mask


# ═══════════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════════

class BinaryCATSAMTrainer:
    """二值 CAT-SAM 训练器 | Binary CAT-SAM trainer.

    模型组合 | Model assembly:
        MobileSAM Encoder (frozen) + CATAdapter + BinaryPromptGenerator
        + SAM MaskDecoder.
    """

    def __init__(self, cfg: dict, args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.args = args
        self.seed = int(cfg.get("seed", 42))
        set_seed(self.seed)
        self.device = torch.device(
            cfg["train"].get("device", "cuda")
            if torch.cuda.is_available()
            else "cpu"
        )
        self._rng = random.Random(self.seed)

        # ── Data ──
        data_root = self._resolve(cfg["data"]["data_root"])
        self.train_ds = NEUSegDataset(root=str(data_root), split=cfg["data"].get("train_split", "train"))
        self.val_ds = NEUSegDataset(root=str(data_root), split=cfg["data"].get("val_split", "test"))
        self.orig_size = tuple(cfg["data"].get("image_size", [200, 200]))
        print(f"[BinaryCATSAM] train={len(self.train_ds)} val={len(self.val_ds)} "
              f"orig_size={self.orig_size}")

        # ── Augmentation ──
        aug_cfg = cfg.get("augmentation", {})
        self.aug = SegAug(
            p_flip=float(aug_cfg.get("p_flip", 0.5)),
            p_rotate=float(aug_cfg.get("p_rotate", 0.5)),
            brightness=float(aug_cfg.get("brightness", 0.2)),
            contrast=float(aug_cfg.get("contrast", 0.2)),
            noise_std=float(aug_cfg.get("noise_std", 0.02)),
        ) if aug_cfg.get("enabled", True) else None

        # ── Model ──
        # Build full SAM once, distribute parts
        ckpt_path = str(self._resolve(cfg["backbone"]["checkpoint"]))
        sam = build_mobile_sam(
            ckpt_path,
            model_type=cfg["backbone"].get("model_type", "vit_t"),
            device=self.device,
        )

        # Backbone: frozen encoder only
        self.backbone = MobileSAMBackbone(sam.image_encoder, img_size=1024).to(self.device)

        # CATAdapter
        adapter_cfg = cfg.get("adapter", {})
        self.adapter = CATAdapter(
            dim=256,
            bottleneck=int(adapter_cfg.get("bottleneck", 64)),
        ).to(self.device)

        # BinaryPromptGenerator
        prompt_cfg = cfg.get("prompt", {})
        self.prompt_generator = BinaryPromptGenerator(
            dim=256,
            bottleneck=int(prompt_cfg.get("bottleneck", 64)),
        ).to(self.device)

        # SAM Decoder
        decoder_cfg_dict = cfg.get("decoder", {})
        self.train_decoder_flag = bool(decoder_cfg_dict.get("train_mask_decoder", False))
        decoder_cfg = SemanticMaskDecoderConfig.from_dict({
            "embed_dim": 256,
            "image_size": 1024,
            "train_mask_decoder": self.train_decoder_flag,
        })
        self.decoder = SemanticMaskDecoder(
            sam.prompt_encoder, sam.mask_decoder, decoder_cfg,
        ).to(self.device)
        # Disable category injection (no support prototype for binary case)
        self.decoder._category_enabled = bool(decoder_cfg_dict.get("category_injection", False))

        # Adapter freeze option
        self.freeze_adapter = getattr(args, "freeze_adapter", False)
        if self.freeze_adapter:
            for p in self.adapter.parameters():
                p.requires_grad_(False)
            print("[BinaryCATSAM] Adapter FROZEN (testing decoder-only upper bound)")

        # ── Trainable params ──
        trainable_params = []
        if not self.freeze_adapter:
            trainable_params.extend(self.adapter.parameters())
        trainable_params.extend(self.prompt_generator.parameters())
        if self.train_decoder_flag:
            trainable_params.extend(
                p for p in self.decoder.mask_decoder.parameters()
                if p.requires_grad
            )
        n_train = sum(p.numel() for p in trainable_params) / 1e6
        n_total = sum(
            p.numel() for m in [
                self.backbone.image_encoder,
                self.adapter,
                self.prompt_generator,
                self.decoder.mask_decoder,
            ] for p in m.parameters()
        ) / 1e6
        print(f"[BinaryCATSAM] trainable={n_train:.2f}M / total={n_total:.2f}M "
              f"(adapter={'frozen' if self.freeze_adapter else 'train'}, "
              f"decoder={'train' if self.train_decoder_flag else 'frozen'})")

        # ── Optimizer ──
        tcfg = cfg["train"]
        self.epochs = int(tcfg.get("epochs", 200))
        self.steps_per_epoch = int(tcfg.get("steps_per_epoch", 200))
        self.batch_size = int(tcfg.get("batch_size", 4))
        self.grad_clip = float(tcfg.get("grad_clip", 1.0))
        self.val_every = int(tcfg.get("val_every", 5))
        lr = float(tcfg.get("lr", 1e-3))
        self.optimizer = AdamW(
            trainable_params,
            lr=lr,
            weight_decay=float(tcfg.get("weight_decay", 1e-4)),
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        # ── Output ──
        exp_parts = [f"binary_catsam_k1_s{self.seed}"]
        if self.freeze_adapter:
            exp_parts.append("frzadapter")
        if self.train_decoder_flag:
            exp_parts.append("trdecoder")
        self.out_dir = self._resolve(cfg.get("output_dir", "runs")) / "_".join(exp_parts)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[BinaryCATSAM] output: {self.out_dir}")

    @staticmethod
    def _resolve(p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else _REPO_ROOT / p

    # ── Data helpers ──

    def _to_binary_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Convert multi-class mask to binary.

        :param mask: [1, H, W] or [H, W] with values {0, 1, 2, 3}.
        :return: [1, H, W] binary float {0, 1}.
        """
        return (mask > 0).float()

    def _preprocess_image(self, img: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Preprocess image for SAM encoder.

        :param img: [3, H, W] float ∈ [0, 1].
        :return: (tensor [3, 1024, 1024], input_size (h, w)).
        """
        x, meta = preprocess_image(img)
        return x, meta.input_size

    # ── Validation ──

    @torch.no_grad()
    def _validate(self) -> dict:
        """Evaluate on validation set.

        Computes:
            - Binary mIoU (BG + FG)
            - FB-IoU (same as FG IoU for binary)
            - Pixel accuracy
            - Per-defect-type IoU (Inclusion, Patch, Scratch vs binary pred)

        :return: metrics dict.
        """
        self.adapter.eval()
        self.prompt_generator.eval()
        if self.train_decoder_flag:
            self.decoder.mask_decoder.eval()

        # Binary metrics
        inter = torch.zeros(2, dtype=torch.float64)  # BG, FG
        union = torch.zeros(2, dtype=torch.float64)
        correct = 0
        total = 0

        # Per-defect-type metrics (binary pred vs each defect type GT)
        defect_names = ["Inclusion", "Patch", "Scratch"]
        defect_inter = {n: 0.0 for n in defect_names}
        defect_union = {n: 0.0 for n in defect_names}
        defect_counts = {n: 0 for n in defect_names}

        for idx in range(len(self.val_ds)):
            sample = self.val_ds[idx]
            img = sample["image"]  # [3, 200, 200] float [0,1]
            gt_multi = sample["masks"].squeeze(0).long()  # [200, 200] {0,1,2,3}
            H, W = gt_multi.shape
            gt_binary = (gt_multi > 0)  # [200, 200] bool

            # Preprocess image
            img_pp, input_size = self._preprocess_image(img)
            img_pp = img_pp.unsqueeze(0).to(self.device)  # [1, 3, 1024, 1024]

            # Forward
            emb = self.backbone(img_pp)["image_embedding"]  # [1, 256, 64, 64]
            adapted = self.adapter(emb)
            sparse, dense = self.prompt_generator(adapted)  # sparse: [1, 256], dense: [1, 256, 64, 64]

            low_res, _ = self.decoder(adapted, sparse, dense)
            # low_res: [1, 1, 256, 256]

            # Upscale to original resolution
            pred_logits = self.decoder.upscale_logits(
                low_res,
                input_size=input_size,
                original_size=(H, W),
            )  # [1, H, W]
            pred_binary = (pred_logits > 0.0).squeeze(0).cpu()  # [H, W] bool

            # ── Binary IoU ──
            for c in range(2):
                pc = pred_binary if c == 1 else ~pred_binary
                gc = gt_binary if c == 1 else ~gt_binary
                inter[c] += (pc & gc).sum().item()  # type: ignore[operator]
                union[c] += (pc | gc).sum().item()  # type: ignore[operator]

            # Pixel accuracy (FG only)
            fg = gt_binary
            correct += (pred_binary[fg] == gt_binary[fg]).sum().item()
            total += fg.sum().item()

            # ── Per-defect-type IoU ──
            for cls_id, name in enumerate(defect_names, start=1):
                gt_cls = (gt_multi == cls_id)
                if gt_cls.sum() > 0:
                    defect_inter[name] += (pred_binary & gt_cls).sum().item()
                    defect_union[name] += (pred_binary | gt_cls).sum().item()
                    defect_counts[name] += 1

        # Compute metrics
        ious = []
        for c in range(2):
            iou = (inter[c] / union[c]).item() if union[c] > 0 else float("nan")
            ious.append(iou)
        miou = float(np.nanmean(ious)) if ious else 0.0
        fg_iou = ious[1] if ious[1] == ious[1] else 0.0  # FB-IoU = FG IoU
        pa = correct / max(total, 1)

        per_defect_iou = {}
        for name in defect_names:
            if defect_counts[name] > 0 and defect_union[name] > 0:
                per_defect_iou[name] = round(
                    defect_inter[name] / defect_union[name], 4
                )
            else:
                per_defect_iou[name] = float("nan")

        # Restore train mode
        self.adapter.train()
        self.prompt_generator.train()
        if self.train_decoder_flag:
            self.decoder.mask_decoder.train()

        return {
            "mIoU": round(miou, 4),
            "FB_IoU": round(fg_iou, 4),
            "BG_IoU": round(ious[0], 4) if ious[0] == ious[0] else 0.0,
            "pixel_accuracy": round(pa, 4),
            "per_defect_IoU": per_defect_iou,
            "n_evaluated": len(self.val_ds),
        }

    # ── Training ──

    def train(self) -> Path:
        """Run training loop.

        :return: path to best checkpoint.
        """
        best_path = self.out_dir / "best_model.pt"
        best_miou = -1.0
        max_steps = getattr(self.args, "steps", None)

        all_indices = list(range(len(self.train_ds)))

        for epoch in range(1, self.epochs + 1):
            self.backbone.eval()  # always frozen
            if not self.freeze_adapter:
                self.adapter.train()
            else:
                self.adapter.eval()
            self.prompt_generator.train()
            if self.train_decoder_flag:
                self.decoder.mask_decoder.train()

            losses = []
            loss_parts = {"bce": [], "focal": [], "dice": []}
            n_steps = 0

            pbar = tqdm(range(self.steps_per_epoch), desc=f"E{epoch:3d}/{self.epochs}")
            for step_i in pbar:
                # ── Sample batch ──
                batch_images_1024 = []
                batch_gt_binary = []
                batch_input_sizes = []
                batch_gt_sizes = []

                for _ in range(self.batch_size):
                    # Rejection sampling: ensure at least one FG pixel
                    for _ in range(50):  # max retries
                        idx = self._rng.randint(0, len(self.train_ds) - 1)
                        sample = self.train_ds[idx]
                        img = sample["image"]  # [3, 200, 200] float
                        gt_multi = sample["masks"]  # [1, 200, 200] int64
                        gt_bin = self._to_binary_mask(gt_multi)  # [1, 200, 200] float

                        # Skip empty (no defect)
                        if gt_bin.sum() < 1:
                            continue
                        break
                    else:
                        # All retries failed (unlikely), use any sample
                        pass

                    H, W = gt_bin.shape[2], gt_bin.shape[1]  # H, W from [1, H, W]

                    # Augmentation
                    if self.aug is not None:
                        img, gt_bin = self.aug(img, gt_bin)

                    # Preprocess image for SAM (resize → normalize → pad)
                    img_pp, input_size = self._preprocess_image(img)
                    batch_images_1024.append(img_pp)
                    batch_gt_binary.append(gt_bin.squeeze(0))  # [H, W]
                    batch_input_sizes.append(input_size)
                    batch_gt_sizes.append((H, W))

                x = torch.stack(batch_images_1024, dim=0).to(self.device)  # [B, 3, 1024, 1024]

                # ── Forward: Encoder + Adapter + PromptGen ──
                with torch.no_grad():
                    emb = self.backbone(x)["image_embedding"]  # [B, 256, 64, 64]

                adapted = self.adapter(emb)  # [B, 256, 64, 64]
                sparse_all, dense_all = self.prompt_generator(adapted)
                # sparse_all: [B, 256], dense_all: [B, 256, 64, 64]

                # ── Decoder: per-sample (decoder expects [1, C, gh, gw]) ──
                total_loss = torch.tensor(0.0, device=self.device)
                loss_detail = {"bce": 0.0, "focal": 0.0, "dice": 0.0}
                valid_samples = 0

                for i in range(self.batch_size):
                    low_res, _iou = self.decoder(
                        adapted[i:i+1],
                        sparse_all[i:i+1],   # [1, C] — decoder expects [1, C]
                        dense_all[i:i+1],
                    )  # low_res: [1, 1, 256, 256]

                    gt_bin_i = batch_gt_binary[i].to(self.device)  # [H, W]

                    # Upscale logits to GT resolution
                    pred_logits = self.decoder.upscale_logits(
                        low_res,
                        input_size=batch_input_sizes[i],
                        original_size=batch_gt_sizes[i],
                    )  # [1, H, W]

                    loss, ld = combined_binary_loss(
                        pred_logits,
                        gt_bin_i.unsqueeze(0),
                        bce_weight=float(self.cfg["loss"].get("bce_weight", 0.3)),
                        focal_weight=float(self.cfg["loss"].get("focal_weight", 0.3)),
                        dice_weight=float(self.cfg["loss"].get("dice_weight", 0.4)),
                        focal_gamma=float(self.cfg["loss"].get("focal_gamma", 3.0)),
                        focal_eps=float(self.cfg["loss"].get("focal_eps", 1e-4)),
                    )

                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    # Scale loss by batch_size for gradient accumulation equivalent
                    total_loss = total_loss + loss / self.batch_size
                    for k in loss_detail:
                        loss_detail[k] += ld[k] / self.batch_size
                    valid_samples += 1

                if valid_samples == 0:
                    continue

                # ── Backward ──
                self.optimizer.zero_grad()
                total_loss.backward()
                # Clip gradients for all trainable modules
                params_to_clip = []
                if not self.freeze_adapter:
                    params_to_clip.extend(self.adapter.parameters())
                params_to_clip.extend(self.prompt_generator.parameters())
                if self.train_decoder_flag:
                    params_to_clip.extend(
                        p for p in self.decoder.mask_decoder.parameters()
                        if p.requires_grad
                    )
                nn.utils.clip_grad_norm_(params_to_clip, self.grad_clip)
                self.optimizer.step()

                losses.append(total_loss.item())
                for k in loss_detail:
                    loss_parts[k].append(loss_detail[k])
                n_steps += 1

                pbar.set_postfix(
                    loss=f"{np.mean(losses[-50:]):.4f}" if losses else "?",
                    bce=f"{np.mean(loss_parts['bce'][-50:]):.4f}" if loss_parts['bce'] else "?",
                )

                if max_steps and n_steps >= max_steps:
                    break

            self.scheduler.step()
            avg_loss = np.mean(losses) if losses else 0.0
            avg_bce = np.mean(loss_parts["bce"]) if loss_parts["bce"] else 0.0
            avg_focal = np.mean(loss_parts["focal"]) if loss_parts["focal"] else 0.0
            avg_dice = np.mean(loss_parts["dice"]) if loss_parts["dice"] else 0.0
            lr = self.optimizer.param_groups[0]["lr"]

            print(f"[BinaryCATSAM] epoch {epoch:>3d} | "
                  f"loss={avg_loss:.4f} (bce={avg_bce:.4f} focal={avg_focal:.4f} dice={avg_dice:.4f}) "
                  f"lr={lr:.2e}")

            # ── Validate ──
            if epoch % self.val_every == 0 or epoch == self.epochs:
                metrics = self._validate()
                miou = metrics["mIoU"]
                is_best = miou > best_miou
                tag = " ★" if is_best else ""
                print(f"[BinaryCATSAM] val | "
                      f"mIoU={metrics['mIoU']:.4f} FB_IoU={metrics['FB_IoU']:.4f} "
                      f"BG_IoU={metrics['BG_IoU']:.4f} PA={metrics['pixel_accuracy']:.4f}{tag}")
                pd_str = " ".join(
                    f"{k}={v:.4f}" for k, v in metrics["per_defect_IoU"].items()
                )
                print(f"[BinaryCATSAM] per_defect: [{pd_str}]")

                if is_best:
                    best_miou = miou
                    self._save(best_path, epoch, avg_loss, metrics)

                # Also log to JSONL
                log_entry = {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "train_bce": avg_bce,
                    "train_focal": avg_focal,
                    "train_dice": avg_dice,
                    **metrics,
                }
                (self.out_dir / "metrics.jsonl").open("a", encoding="utf-8").write(
                    json.dumps(log_entry, ensure_ascii=False) + "\n"
                )

            if max_steps and n_steps >= max_steps:
                break

        # ── Save last ──
        if best_miou < 0:
            # Never validated (e.g., 1 epoch with val_every > 1)
            metrics = self._validate()
            best_miou = metrics["mIoU"]
            self._save(best_path, self.epochs, avg_loss, metrics)

        self._save(self.out_dir / "last_model.pt", self.epochs, avg_loss,
                   getattr(self, "_last_metrics", None))
        print(f"[BinaryCATSAM] done. best_mIoU={best_miou:.4f} → {best_path}")
        return best_path

    def _save(
        self, path: Path, epoch: int, loss: float, metrics: dict | None = None
    ) -> None:
        """Save checkpoint."""
        data: dict = {
            "epoch": epoch,
            "stage": "binary_catsam",
            "adapter": self.adapter.state_dict(),
            "prompt_generator": self.prompt_generator.state_dict(),
            "loss": loss,
            "config": self.cfg,
            "args": vars(self.args),
            "train_decoder": self.train_decoder_flag,
            "freeze_adapter": self.freeze_adapter,
        }
        # Save mask_decoder state if it was trained
        if self.train_decoder_flag:
            data["mask_decoder"] = self.decoder.mask_decoder.state_dict()
        if metrics:
            data["metrics"] = metrics
        torch.save(data, path)
        self._last_metrics = metrics
        # Save a human-readable summary
        summary = {
            "epoch": epoch,
            "loss": round(loss, 6),
            "train_decoder": self.train_decoder_flag,
            "freeze_adapter": self.freeze_adapter,
        }
        if metrics:
            summary.update({k: v for k, v in metrics.items() if k != "per_defect_IoU"})
            summary["per_defect_IoU"] = metrics.get("per_defect_IoU", {})
        (self.out_dir / "last_metrics.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NEU-SEG Binary CAT-SAM Training | 二值化 CAT-SAM 训练"
    )
    p.add_argument(
        "--config", default=str(_REPO_ROOT / "configs" / "neu_seg_binary.yaml"),
        help="Path to YAML config file",
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--steps", type=int, default=None,
                   help="Limit steps per epoch (smoke test)")
    p.add_argument("--steps-per-epoch", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--weights", default=None,
                   help="MobileSAM weights path override")
    p.add_argument("--train-decoder", action="store_true", default=None,
                   help="Train SAM MaskDecoder (default: frozen)")
    p.add_argument("--freeze-adapter", action="store_true", default=None,
                   help="Freeze CATAdapter (test decoder-only upper bound)")
    p.add_argument("--no-aug", action="store_true", default=None)
    return p.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    overrides = [
        (("train", "epochs"), args.epochs),
        (("train", "steps_per_epoch"), args.steps_per_epoch),
        (("train", "batch_size"), args.batch_size),
        (("train", "lr"), args.lr),
        (("train", "device"), args.device),
        (("seed",), args.seed),
        (("output_dir",), args.output_dir),
        (("data", "data_root"), args.data_root),
    ]
    for keys, val in overrides:
        if val is not None:
            d = cfg
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = val

    if args.train_decoder is not None:
        cfg.setdefault("decoder", {})["train_mask_decoder"] = args.train_decoder
    if args.weights is not None:
        cfg.setdefault("backbone", {})["checkpoint"] = args.weights
    if args.no_aug is not None:
        cfg.setdefault("augmentation", {})["enabled"] = False

    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    trainer = BinaryCATSAMTrainer(cfg, args)
    best = trainer.train()
    print(f"\n[train_binary_catsam] done. best: {best}")


if __name__ == "__main__":
    main()
