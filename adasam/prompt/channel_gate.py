"""
可学习通道门控 | Learnable Channel Gate.
=========================================

在 PromptFusion 输出后插入逐通道可学习门控, 配合 L1 稀疏正则,
迫使网络显式选择有用通道, 而非依赖 Decoder 隐式筛选。

Inserts a per-channel learnable gate after PromptFusion output,
with L1 sparsity regularization to force explicit channel selection
instead of relying on the Decoder's implicit filtering.

设计 | Design:
    - gate_logits: [C] 可学习参数, 初始化为 gate_init (默认 2.0 → sigmoid≈0.88)
    - 前向: x = gate · x  (gate = sigmoid(gate_logits))
    - L1 损失: λ·mean(gate) 推动不必要通道趋近于 0
    - 有用通道被子任务的梯度"保护" (self-balancing sparsity)

参考 | Reference:
    - Learned Sparsity via L1 Regularization (Ng 2004)
    - Channel Gating Networks (Hua et al. 2019)
    - Hard-Concrete / L0 Regularization (Louizos et al. 2018)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelGate(nn.Module):
    """可学习逐通道门控 | Learnable per-channel gate.

    :param num_channels: 通道数 | Number of channels (default 256).
    :param init_gate: gate_logits 初始值 | Initial logit value.
        Higher = more active at start. 2.0 → sigmoid≈0.88.
    :param hard_threshold: 推理时硬阈值 (None=soft gate, 0.5=hard binary).
        Hard threshold for inference (None=soft, 0.5=binary).
    """

    def __init__(
        self,
        num_channels: int = 256,
        init_gate: float = 2.0,
        hard_threshold: float | None = None,
    ) -> None:
        super().__init__()
        self.gate_logits = nn.Parameter(
            torch.full((num_channels,), init_gate)
        )
        self.hard_threshold = hard_threshold

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """应用门控 | Apply gating.

        :param x: [B, C, H, W] dense_prompt.
        :return: (gated_x [B, C, H, W], gate_probs [C]).
        """
        gate = self.gate_values()  # [C] in (0, 1)

        if self.training:
            gate_out = gate
        elif self.hard_threshold is not None:
            gate_out = (gate > self.hard_threshold).float()
        else:
            gate_out = gate

        return x * gate_out.view(1, -1, 1, 1), gate

    def gate_values(self) -> torch.Tensor:
        """返回门控概率 | Return gate probabilities [C]."""
        return torch.sigmoid(self.gate_logits)

    # ── Monitoring helpers ──

    def sparsity(self) -> torch.Tensor:
        """通道稀疏度 (gate<0.1 的比例) | Fraction of channels with gate < 0.1."""
        return (self.gate_values() < 0.1).float().mean()

    def n_active(self, threshold: float = 0.5) -> torch.Tensor:
        """活跃通道数 | Number of active channels (gate > threshold)."""
        return (self.gate_values() > threshold).sum()

    def gate_stats(self) -> dict[str, float]:
        """门控统计 (用于监控) | Gate statistics for monitoring."""
        gv = self.gate_values().detach()
        return {
            "gate_mean": float(gv.mean()),
            "gate_std": float(gv.std()),
            "gate_min": float(gv.min()),
            "gate_max": float(gv.max()),
            "gate_sparsity": float((gv < 0.1).float().mean()),
            "gate_n_active": int((gv > 0.5).sum()),
            "gate_n_active_loose": int((gv > 0.1).sum()),
        }
