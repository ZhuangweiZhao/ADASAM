"""
CAT-SAM 适配器模块 | CAT-SAM adapter modules.
==============================================

CAT-SAM 启发的轻量特征适配器, 用于弥合 Natural Images → Remote Sensing 域差异。
CAT-SAM-inspired lightweight feature adapter for bridging the domain gap between
natural images and remote sensing imagery.

导出 | Exports:
    CATAdapter — 瓶颈残差特征适配器 | bottleneck residual feature adapter.
"""

from adasam.adapters.cat_adapter import CATAdapter, MultiScaleCATAdapter

__all__ = ["CATAdapter", "MultiScaleCATAdapter"]
from adasam.adapters.lora import LoRALinear, inject_tinyvit_lora
