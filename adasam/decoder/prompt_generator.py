"""
Learnable Prompt Generator | 可学习提示生成器.
================================================

将候选区域 ( CandidateSet ) 转化为丰富的 SAM 提示:
    point_xy, box_xyxy, prompt_token, region_score

Converts candidate regions (CandidateSet) into rich SAM prompts:
    point_xy, box_xyxy, prompt_token, region_score

核心设计 | Core design:
    - Point / Box: 从候选直接传递 (几何量, 不需要学习)。
      Point / Box: passed through from candidate (geometric, no learning).
    - Prompt Token: 残差形式 prototype + Δprompt, Δprompt 零初始化。
      Prompt Token: residual form prototype + Δprompt, Δprompt zero-initialized.
      训练学习的是原型在每个 region 上的调整量。
      Training learns the per-region adjustment to the prototype.
    - Region Score: 可学习置信度 (MLP → Sigmoid), 替代启发式 region_score_raw。
      Region Score: learnable confidence (MLP → Sigmoid), replaces heuristic score.

输入 | Input (per region):
    - prototype [256]          全局类原型 | global class prototype
    - query_feature [256]      候选区域池化 query 特征 | pooled query feature
    - per_support_sim [K]      每 support 相似度 (仅用于提取统计量 | only for stats)
    - boxes [N,4] + scores_raw [N]  (仅用于提取几何特征 | only for geometric features)

输出 | Output (per region):
    - point_xy [2]             (传递 | passthrough)
    - box_xyxy [4]             (传递 | passthrough)
    - prompt_token [256]        prototype + Δprompt
    - region_score [1]          可学习置信度 ∈ [0, 1]

K-无关设计 | K-invariant design:
    per_support_sim [K] 被压缩为 3 个统计量 (mean, max, std),
    与 K 个几何特征拼接 → 输入维度恒定, 适用于任意 K。
    per_support_sim [K] is compressed to 3 statistics (mean, max, std),
    concatenated with geometric features → constant input dim for any K.

参数量 | Parameters: ~200K (input_proj: 519→256, delta_prompt, score_head).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptGenerator(nn.Module):
    """可学习提示生成器 | Learnable Prompt Generator.

    :param embed_dim: 嵌入维度 (256) | embedding dim.
    :param hidden_dim: MLP 隐藏维 | MLP hidden dim.
    :param n_geo_feats: 几何特征数 (不含 sim 统计量) | geometric feature count (excl. sim stats).
    :param n_sim_stats: 跨 support 统计量数 (mean, max, std = 3) | cross-support stat count.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        n_geo_feats: int = 4,
        n_sim_stats: int = 3,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Input: prototype [C] + query_feature [C] + sim_stats [3] + geo [4] = 2C + 7
        input_dim = embed_dim * 2 + n_sim_stats + n_geo_feats

        # Shared input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # ── Residual prompt token branch ──
        # Δprompt = MLP(hidden → hidden → embed_dim), zero-init final layer
        self.delta_prompt = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        # Zero-init → at epoch 0, prompt_token = prototype (identity behavior)
        nn.init.zeros_(self.delta_prompt[-1].weight)
        nn.init.zeros_(self.delta_prompt[-1].bias)

        # ── Region score branch ──
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def _compute_features(
        self,
        boxes: torch.Tensor,
        per_support_sim: torch.Tensor,
        scores_raw: torch.Tensor,
        image_size: float = 1024.0,
    ) -> torch.Tensor:
        """从候选属性提取归一化特征 | Extract normalized features from candidate attributes.

        几何特征 (4 个) | Geometric features (4):
            1. area_norm     — 面积 / 图像面积 | area / image_area
            2. aspect_ratio  — 宽高比 (clamped) | aspect ratio
            3. score_raw     — 启发式 region 得分 | heuristic region score
            4. n_active      — 活跃 support 比例 | fraction of active supports

        跨 support 统计量 (3 个) | Cross-support statistics (3):
            5. sim_mean      — per_support_sim 均值 | mean across supports
            6. sim_max       — per_support_sim 最大值 | max across supports
            7. sim_std       — per_support_sim 标准差 | std across supports

        :param boxes: [N, 4] bbox xyxy in input frame.
        :param per_support_sim: [N, K] per-support mean similarity.
        :param scores_raw: [N] region_score_raw ∈ [0, 1].
        :param image_size: 输入帧边长 | input frame side length.
        :return: [N, 7] features (4 geo + 3 sim stats).
        """
        N = boxes.shape[0]
        w = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0)   # [N]
        h = (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0)   # [N]

        # ── Geometric features ──
        area_norm = (w * h) / (image_size * image_size)     # [N]
        aspect = (w / h.clamp(min=1.0)).clamp(0.1, 10.0)   # [N]
        score_raw = scores_raw                               # [N]
        n_active = (per_support_sim > 0.0).float().mean(dim=1)  # [N]

        # ── Cross-support sim statistics (K-invariant!) ──
        sim_mean = per_support_sim.mean(dim=1)   # [N]
        sim_max = per_support_sim.max(dim=1).values   # [N]
        sim_std = per_support_sim.std(dim=1)     # [N]

        return torch.stack([
            area_norm, aspect, score_raw, n_active,
            sim_mean, sim_max, sim_std,
        ], dim=1)  # [N, 7]

    def forward(
        self,
        prototype: torch.Tensor,
        candidate_coords: torch.Tensor,
        candidate_boxes: torch.Tensor,
        candidate_query_features: torch.Tensor,
        candidate_per_support_sim: torch.Tensor,
        candidate_scores_raw: torch.Tensor,
        image_size: float = 1024.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """为一批候选生成提示 | Generate prompts for a batch of candidates.

        :param prototype: [C] global class prototype (单个 | single).
        :param candidate_coords: [N, 2] candidate centroids.
        :param candidate_boxes: [N, 4] candidate bboxes (xyxy).
        :param candidate_query_features: [N, C] pooled query features.
        :param candidate_per_support_sim: [N, K] per-support mean similarity (任意 K | any K).
        :param candidate_scores_raw: [N] heuristic region scores.
        :param image_size: input frame side length (for area normalization).
        :return: (point_xy [N,2], box_xyxy [N,4], prompt_token [N,C], region_score [N,1]).
        """
        N = candidate_coords.shape[0]
        C = self.embed_dim

        # Expand prototype to batch
        proto_batch = prototype.unsqueeze(0).expand(N, -1)  # [N, C]

        # Compute K-invariant features (4 geo + 3 sim stats = 7)
        features = self._compute_features(
            candidate_boxes, candidate_per_support_sim, candidate_scores_raw, image_size,
        )  # [N, 7]

        # ── Fuse all inputs ──
        x = torch.cat([
            proto_batch,                # [N, C]
            candidate_query_features,    # [N, C]
            features,                    # [N, 7]
        ], dim=-1)  # [N, 2C + 7] = [N, 519]

        x = F.gelu(self.input_proj(x))  # [N, hidden_dim]

        # ── Residual prompt token ──
        delta = self.delta_prompt(x)                       # [N, C]
        prompt_token = proto_batch + delta                 # residual: prototype + Δ

        # ── Region score ──
        region_score = self.score_head(x)                  # [N, 1]

        # Point and box pass through unchanged
        return candidate_coords, candidate_boxes, prompt_token, region_score
