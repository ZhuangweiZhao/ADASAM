"""Standard baselines for the label-efficient semantic-segmentation protocol.

Provides two widely cited baselines with the same interface as the proposed
models (``forward(image) -> logits`` at the input resolution and
``parameter_counts()``):

- ``DeepLabV3PlusBaseline``: DeepLabV3+ via segmentation-models-pytorch
  (ResNet encoder, ImageNet pretrained when requested).
- ``SegFormerBaseline``: SegFormer (MixVisionTransformer + MLP decode head).
  The MIT backbone is a self-contained implementation matching the official
  NVIDIA/SegFormer ``util/mit.py`` semantics; ImageNet-1k pretrained weights are
  converted from the HuggingFace ``nvidia/mit-b{0,1,2}`` checkpoints by
  ``tools/setup_segformer_weights.py``.

Inputs are RGB images in ``[0, 1]`` (the dataset convention); both models
internally apply ImageNet normalization, so the training loop is unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from adasam.utils.transforms import PIXEL_MEAN, PIXEL_STD

SEGFORMER_CONFIGS: dict[str, dict] = {
    "b0": {
        "depths": [2, 2, 2, 2],
        "embed_dims": [32, 64, 160, 256],
        "num_heads": [1, 2, 5, 8],
        "sr_ratios": [8, 4, 2, 1],
        "mlp_ratio": 4,
    },
    "b1": {
        "depths": [2, 2, 2, 2],
        "embed_dims": [64, 128, 320, 512],
        "num_heads": [1, 2, 5, 8],
        "sr_ratios": [8, 4, 2, 1],
        "mlp_ratio": 4,
    },
    "b2": {
        "depths": [3, 4, 6, 3],
        "embed_dims": [64, 128, 320, 512],
        "num_heads": [1, 2, 5, 8],
        "sr_ratios": [8, 4, 2, 1],
        "mlp_ratio": 4,
    },
}

SEGFORMER_WEIGHTS = {
    "b0": "mit_b0.pth",
    "b1": "mit_b1.pth",
    "b2": "mit_b2.pth",
}


# ───────────────────────── DeepLabV3+ ─────────────────────────


class DeepLabV3PlusBaseline(nn.Module):
    """DeepLabV3+ (ResNet encoder) via segmentation-models-pytorch."""

    def __init__(self, num_classes: int, encoder_name: str = "resnet50",
                 pretrained: bool = True) -> None:
        super().__init__()
        try:
            import segmentation_models_pytorch as smp  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "DeepLabV3PlusBaseline requires segmentation_models_pytorch; "
                "install it with: pip install segmentation_models_pytorch"
            ) from exc
        self.encoder_name = encoder_name
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=num_classes,
        )

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(PIXEL_MEAN, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        std = torch.tensor(PIXEL_STD, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        return (image * 255.0 - mean) / std

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"expected image [B,3,H,W], got {tuple(image.shape)}")
        # DeepLabV3+ (ResNet encoder) requires dimensions divisible by 16.
        height, width = image.shape[-2:]
        pad_h = (16 - height % 16) % 16
        pad_w = (16 - width % 16) % 16
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
        logits = self.model(self._preprocess(image))
        if pad_h or pad_w:
            logits = logits[..., :height, :width]
        return logits

    def forward_with_auxiliary(
        self, image: torch.Tensor, target: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, None, None]:
        """Interface parity with LabelEfficientSAM; baselines have no auxiliary outputs."""
        return self.forward(image), None, None

    def parameter_counts(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": total, "frozen": 0}


# ───────────────────────── SegFormer ─────────────────────────


def _to_2tuple(value: int | tuple[int, int]) -> tuple[int, int]:
    return (value, value) if isinstance(value, int) else value


def _trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 0.02) -> torch.Tensor:
    with torch.no_grad():
        if std > 0:
            bound = 2.0 * std
            tensor.uniform_(-bound, bound)
        else:
            tensor.zero_()
    return tensor


class DropPath(nn.Module):
    """Stochastic depth (per-sample drop)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device) < keep_prob
        return x * mask / keep_prob


class DWConv(nn.Module):
    """Official SegFormer MixFFN depthwise conv over the flattened sequence."""

    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C] -> [B, C, N, 1] -> conv over N -> back
        x = x.permute(0, 2, 1).contiguous().unsqueeze(-1)
        x = self.dwconv(x).squeeze(-1).permute(0, 2, 1).contiguous()
        return x


class Mlp(nn.Module):
    """MixFFN: fc1 -> dwconv -> act -> fc2."""

    def __init__(self, in_features: int, hidden_features: int | None = None,
                 out_features: int | None = None, act_layer: type[nn.Module] = nn.GELU,
                 drop: float = 0.0) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Spatial-reduction attention (q + kv projections, no position bias)."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 qk_scale: float | None = None, attn_drop: float = 0.0,
                 proj_drop: float = 0.0, sr_ratio: int = 1) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, num_tokens, dim = x.shape
        heads = self.num_heads
        head_dim = dim // heads
        q = self.q(x).reshape(batch, num_tokens, heads, head_dim).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            height = width = int(math.sqrt(num_tokens))
            x_ = x.permute(0, 2, 1).reshape(batch, dim, height, width)
            x_ = self.sr(x_).reshape(batch, dim, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(batch, -1, 2, heads, head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(batch, -1, 2, heads, head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch, num_tokens, dim)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, qk_scale: float | None = None,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0, act_layer: type[nn.Module] = nn.GELU,
                 norm_layer: type[nn.Module] = nn.LayerNorm, sr_ratio: int = 1) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer, drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 7, stride: int = 4, in_chans: int = 3,
                 embed_dim: int = 768, norm_layer: type[nn.Module] = nn.LayerNorm) -> None:
        super().__init__()
        patch_size = _to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = norm_layer(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, C, H, W]
        batch, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return self.norm(x), (height, width)


class MixVisionTransformer(nn.Module):
    """Self-contained SegFormer backbone (official NVIDIA mit.py semantics)."""

    def __init__(self, variant: str = "b0", img_size: int = 224,
                 in_chans: int = 3, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0, drop_path_rate: float = 0.1,
                 qkv_bias: bool = True) -> None:
        super().__init__()
        if variant not in SEGFORMER_CONFIGS:
            raise ValueError(f"variant must be one of {sorted(SEGFORMER_CONFIGS)}, got {variant!r}")
        cfg = SEGFORMER_CONFIGS[variant]
        depths = cfg["depths"]
        embed_dims = cfg["embed_dims"]
        num_heads = cfg["num_heads"]
        sr_ratios = cfg["sr_ratios"]
        mlp_ratio = cfg["mlp_ratio"]
        self.variant = variant
        self.embed_dims = embed_dims
        self.depths = depths

        norm_layer = nn.LayerNorm
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.patch_embed1 = OverlapPatchEmbed(
            patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0], norm_layer=norm_layer)
        self.patch_embed2 = OverlapPatchEmbed(
            patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1], norm_layer=norm_layer)
        self.patch_embed3 = OverlapPatchEmbed(
            patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2], norm_layer=norm_layer)
        self.patch_embed4 = OverlapPatchEmbed(
            patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3], norm_layer=norm_layer)

        self.pos_drop = nn.Dropout(p=drop_rate)
        block_index = 0
        self.block1 = nn.Sequential(*[
            Block(embed_dims[0], num_heads[0], mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_index + i],
                  norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for i in range(depths[0])
        ])
        block_index += depths[0]
        self.block2 = nn.Sequential(*[
            Block(embed_dims[1], num_heads[1], mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_index + i],
                  norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for i in range(depths[1])
        ])
        block_index += depths[1]
        self.block3 = nn.Sequential(*[
            Block(embed_dims[2], num_heads[2], mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_index + i],
                  norm_layer=norm_layer, sr_ratio=sr_ratios[2])
            for i in range(depths[2])
        ])
        block_index += depths[2]
        self.block4 = nn.Sequential(*[
            Block(embed_dims[3], num_heads[3], mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_index + i],
                  norm_layer=norm_layer, sr_ratio=sr_ratios[3])
            for i in range(depths[3])
        ])
        self.norm1 = norm_layer(embed_dims[0])
        self.norm2 = norm_layer(embed_dims[1])
        self.norm3 = norm_layer(embed_dims[2])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            _trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            _trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return 4 stage feature maps at 1/4..1/32 resolution as [B, C, H, W]."""
        batch = x.shape[0]
        outs: list[torch.Tensor] = []

        x, (h1, w1) = self.patch_embed1(x)
        x = self.pos_drop(x)
        x = self.block1(x)
        x = self.norm1(x)
        x = x.reshape(batch, h1, w1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, (h2, w2) = self.patch_embed2(x)
        x = self.block2(x)
        x = self.norm2(x)
        x = x.reshape(batch, h2, w2, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, (h3, w3) = self.patch_embed3(x)
        x = self.block3(x)
        x = self.norm3(x)
        x = x.reshape(batch, h3, w3, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, (h4, w4) = self.patch_embed4(x)
        x = self.block4(x)
        x = self.norm4(x)
        x = x.reshape(batch, h4, w4, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        return outs


class SegFormerHead(nn.Module):
    """MLP decode head (mmseg-style SegformerHead)."""

    def __init__(self, num_classes: int, in_channels: list[int],
                 channels: int = 256, dropout_ratio: float = 0.1) -> None:
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, channels, kernel_size=1) for in_ch in in_channels
        ])
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * len(in_channels), channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = inputs
        target_size = c1.shape[-2:]
        x1 = self.convs[0](c1)
        x2 = F.interpolate(self.convs[1](c2), size=target_size, mode="bilinear", align_corners=False)
        x3 = F.interpolate(self.convs[2](c3), size=target_size, mode="bilinear", align_corners=False)
        x4 = F.interpolate(self.convs[3](c4), size=target_size, mode="bilinear", align_corners=False)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.fusion_conv(x)
        x = self.dropout(x)
        return self.cls_seg(x)


class SegFormerBaseline(nn.Module):
    """SegFormer (MIT-B0/B1/B2) for semantic segmentation at input resolution."""

    def __init__(self, num_classes: int, variant: str = "b0",
                 pretrained: bool = True, weights_root: str | Path = "weights") -> None:
        super().__init__()
        self.variant = variant
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.backbone = MixVisionTransformer(variant=variant)
        self.head = SegFormerHead(
            num_classes=num_classes, in_channels=self.backbone.embed_dims
        )
        if pretrained:
            path = Path(weights_root) / SEGFORMER_WEIGHTS[variant]
            if not path.exists():
                raise FileNotFoundError(
                    f"SegFormer-{variant.upper()} pretrained weights not found at {path}; "
                    f"run `python tools/setup_segformer_weights.py` first, "
                    f"or pass pretrained=False for random initialization."
                )
            self._load_weights(path)

    def _load_weights(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        state = {k: v for k, v in state.items() if not k.startswith("head.")}
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys in {path}: {unexpected[:5]} ...")
        # Pretrained backbone loads into the frozen-partition accounting as trainable
        # by design (baselines are fully fine-tuned, matching the protocol).

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(PIXEL_MEAN, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        std = torch.tensor(PIXEL_STD, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        return (image * 255.0 - mean) / std

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"expected image [B,3,H,W], got {tuple(image.shape)}")
        output_size = tuple(image.shape[-2:])
        features = self.backbone(self._preprocess(image))
        logits = self.head(features)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

    def forward_with_auxiliary(
        self, image: torch.Tensor, target: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, None, None]:
        """Interface parity with LabelEfficientSAM; baselines have no auxiliary outputs."""
        return self.forward(image), None, None

    def parameter_counts(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": total, "frozen": 0}


# ───────────────────────── factory ─────────────────────────


def build_baseline(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    encoder_name: str = "resnet50",
    segformer_variant: str = "b0",
    weights_root: str | Path = "weights",
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Build a standard baseline model by name."""
    if name == "deeplabv3plus":
        model = DeepLabV3PlusBaseline(
            num_classes=num_classes, encoder_name=encoder_name, pretrained=pretrained
        )
    elif name == "segformer":
        model = SegFormerBaseline(
            num_classes=num_classes, variant=segformer_variant,
            pretrained=pretrained, weights_root=weights_root,
        )
    else:
        raise ValueError(f"unknown baseline {name!r}; expected deeplabv3plus|segformer")
    return model.to(device)
