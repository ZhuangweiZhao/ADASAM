"""Compact U-Net for the common label-efficient semantic protocol."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )


class LabelEfficientUNet(nn.Module):
    def __init__(self, num_classes: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        b = base_channels
        self.enc1 = _DoubleConv(3, b)
        self.enc2 = _DoubleConv(b, b * 2)
        self.enc3 = _DoubleConv(b * 2, b * 4)
        self.bottleneck = _DoubleConv(b * 4, b * 8)
        self.dec3 = _DoubleConv(b * 8 + b * 4, b * 4)
        self.dec2 = _DoubleConv(b * 4 + b * 2, b * 2)
        self.dec1 = _DoubleConv(b * 2 + b, b)
        self.classifier = nn.Conv2d(b, num_classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(image)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        x = self.bottleneck(F.max_pool2d(e3, 2))
        x = F.interpolate(x, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec3(torch.cat([x, e3], dim=1))
        x = F.interpolate(x, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, e2], dim=1))
        x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self.classifier(self.dec1(torch.cat([x, e1], dim=1)))

    def parameter_counts(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": total, "frozen": 0}
