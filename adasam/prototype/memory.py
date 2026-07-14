"""
原型记忆 | Prototype Memory.
============================

按类别存储原型的极简容器: 支持累加 (running mean) 与查询。
Minimal per-class prototype store: supports accumulation (running mean) and lookup.

用途 | Usage:
    - 评估阶段: 为每个类从其 K-shot support 构建一次原型并 add()。
      Eval: build one prototype per class from its K-shot support and add() it.
    - 训练阶段 (可选): 跨 episode 累加同类原型以稳定表示。
      Training (optional): accumulate same-class prototypes across episodes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class PrototypeMemory:
    """按类别的原型记忆 | Per-class prototype memory.

    :param embed_dim: 原型维度 | prototype dim (256).
    """

    def __init__(self, embed_dim: int = 256) -> None:
        self.embed_dim = embed_dim
        self._protos: dict[int, torch.Tensor] = {}
        self._counts: dict[int, int] = {}

    def add(self, class_id: int, prototype: torch.Tensor) -> None:
        """累加一个类原型 (running mean, 归一化存储) | Accumulate a prototype (running mean, normalized).

        :param class_id: 类别 ID | class ID.
        :param prototype: [embed_dim] 原型向量 | prototype vector.
        """
        if prototype.shape != (self.embed_dim,):
            raise ValueError(f"expected [{self.embed_dim}], got {tuple(prototype.shape)}")
        p = prototype.detach()
        if class_id not in self._protos:
            self._protos[class_id] = p.clone()
            self._counts[class_id] = 1
        else:
            n = self._counts[class_id]
            self._protos[class_id] = (self._protos[class_id] * n + p) / (n + 1)
            self._counts[class_id] = n + 1
        self._protos[class_id] = F.normalize(self._protos[class_id], dim=0)

    def get(self, class_id: int) -> torch.Tensor:
        """取类原型 | Get a class prototype.

        :raises KeyError: 若该类未登记 | if the class is absent.
        """
        if class_id not in self._protos:
            raise KeyError(f"no prototype for class {class_id}")
        return self._protos[class_id]

    def has(self, class_id: int) -> bool:
        """是否已登记该类 | whether the class is present."""
        return class_id in self._protos

    def classes(self) -> list[int]:
        """已登记类别 ID | registered class IDs."""
        return sorted(self._protos)

    def clear(self) -> None:
        """清空 | reset."""
        self._protos.clear()
        self._counts.clear()
