"""Lightweight FPN decoder for label-efficient semantic segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adasam.adapters import CATAdapter


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )


class LightweightSemanticDecoder(nn.Module):
    """Fuse P3, P4 and MobileSAM embedding features with a compact FPN."""

    def __init__(
        self,
        num_classes: int,
        feature_dims: dict[str, int] | None = None,
        decoder_dim: int = 96,
        enable_prompt_fusion: bool = False,
        enable_spatial_prompt_fusion: bool = False,
        spatial_prompt_mode: str = "both",
        feature_scales: str = "p3_p4_embedding",
        fusion_version: str = "hierarchical",
        representation_budget: int = 3,
        spatial_policy: str = "adaptive",
        feature_retention_ratio: float = 1.0,
        spatial_budget_temperature: float = 1.0,
        static_importance_map: torch.Tensor | None = None,
        post_fusion_adapter: bool = False,
        adapter_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        if spatial_prompt_mode not in {"both", "dense", "token"}:
            raise ValueError("spatial_prompt_mode must be one of: both, dense, token")
        scale_names = {
            "p3": ("P3",), "p4": ("P4",),
            "embedding": ("embedding",),
            "p3_p4": ("P3", "P4"),
            "p3_embedding": ("P3", "embedding"),
            "p4_embedding": ("P4", "embedding"),
            "p3_p4_embedding": ("P3", "P4", "embedding"),
        }
        if feature_scales not in scale_names:
            raise ValueError(
                "feature_scales must be one of: p3, p4, embedding, p3_p4, p3_embedding, p4_embedding, p3_p4_embedding"
            )
        if fusion_version not in {
            "hierarchical", "concat", "sum", "global", "image_conditioned",
            "scsr", "scsr_v2", "scsr_task", "semantic_budget",
        }:
            raise ValueError(
                "fusion_version must be hierarchical, concat, global, "
                "sum, image_conditioned, scsr, scsr_v2, scsr_task, or semantic_budget"
            )
        if fusion_version in {"scsr", "scsr_v2", "scsr_task"} and feature_scales != "p3_p4_embedding":
            raise ValueError("SCSR requires feature_scales=p3_p4_embedding")
        if fusion_version == "semantic_budget" and feature_scales != "p3_p4_embedding":
            raise ValueError("semantic_budget requires feature_scales=p3_p4_embedding")
        if representation_budget not in {1, 2, 3}:
            raise ValueError("representation_budget must be 1, 2, or 3")
        if spatial_policy not in {"adaptive", "static", "magnitude", "distilled_magnitude", "random"}:
            raise ValueError(
                "spatial_policy must be adaptive, static, magnitude, distilled_magnitude, or random"
            )
        if not 0.0 < feature_retention_ratio <= 1.0:
            raise ValueError("feature_retention_ratio must be in (0, 1]")
        if spatial_budget_temperature <= 0.0:
            raise ValueError("spatial_budget_temperature must be positive")
        self.feature_scales = feature_scales
        self.feature_names = scale_names[feature_scales]
        self.fusion_version = fusion_version
        self.representation_budget = representation_budget
        self.spatial_policy = spatial_policy
        self.feature_retention_ratio = feature_retention_ratio
        self.spatial_budget_temperature = spatial_budget_temperature
        dims = feature_dims or {"P3": 128, "P4": 160, "embedding": 256}
        self.lateral = nn.ModuleDict(
            {
                name: nn.Conv2d(dims[name], decoder_dim, 1)
                for name in self.feature_names
            }
        )
        self.p4_fuse = (
            ConvNormAct(decoder_dim, decoder_dim) if "P4" in self.feature_names else None
        )
        self.p3_fuse = (
            ConvNormAct(decoder_dim, decoder_dim) if "P3" in self.feature_names else None
        )
        self.concat_projection = nn.Conv2d(decoder_dim * len(self.feature_names), decoder_dim, 1) if fusion_version == "concat" else None
        self.global_logits = nn.Parameter(torch.zeros(len(self.feature_names))) if fusion_version == "global" else None
        self.image_router = nn.Linear(decoder_dim, len(self.feature_names)) if fusion_version == "image_conditioned" else None
        self.scsr_gamma = nn.Parameter(torch.zeros(2)) if fusion_version == "scsr" else None
        self.scsr_beta = nn.Parameter(torch.zeros(3)) if fusion_version == "scsr" else None
        self.scsr_confidence = nn.Conv2d(decoder_dim, 1, 1) if fusion_version == "scsr" else None
        if self.scsr_confidence is not None:
            nn.init.zeros_(self.scsr_confidence.weight)
            nn.init.zeros_(self.scsr_confidence.bias)
        # SCSR-v2 scores all scales with one symmetric local router. Its input
        # contains the aligned feature, semantic anchor, complementarity map,
        # cosine compatibility, and mean absolute feature difference.
        self.scsr_v2_router = (
            nn.Conv2d(decoder_dim * 3 + 2, 1, 1)
            if fusion_version in {"scsr_v2", "scsr_task"} else None
        )
        self.scsr_v2_bias = (
            nn.Parameter(torch.zeros(3))
            if fusion_version in {"scsr_v2", "scsr_task"} else None
        )
        self.scsr_v2_log_temperature = (
            nn.Parameter(torch.zeros(()))
            if fusion_version in {"scsr_v2", "scsr_task"} else None
        )
        if self.scsr_v2_router is not None:
            nn.init.zeros_(self.scsr_v2_router.weight)
            nn.init.zeros_(self.scsr_v2_router.bias)
        self.scsr_task_heads = (
            nn.ModuleList(nn.Conv2d(decoder_dim, num_classes, 1) for _ in range(3))
            if fusion_version == "scsr_task" else None
        )
        if self.scsr_task_heads is not None:
            for head in self.scsr_task_heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        self.task_routing_hard = False
        self.semantic_budget_controller = (
            nn.Linear(decoder_dim, 2) if fusion_version == "semantic_budget" else None
        )
        self.semantic_budget_gate = (
            nn.Conv2d(decoder_dim, 1, 1, bias=True) if fusion_version == "semantic_budget" else None
        )
        self.semantic_budget_residual = (
            nn.Parameter(torch.zeros(2)) if fusion_version == "semantic_budget" else None
        )
        self.semantic_budget_student = (
            nn.Sequential(
                nn.Conv2d(
                    decoder_dim, decoder_dim, 3, padding=1,
                    groups=decoder_dim, bias=False,
                ),
                nn.GroupNorm(8, decoder_dim),
                nn.GELU(),
                nn.Conv2d(decoder_dim, 2, 1),
            )
            if fusion_version == "semantic_budget" and spatial_policy == "distilled_magnitude"
            else None
        )
        if static_importance_map is not None:
            if fusion_version != "semantic_budget" or spatial_policy != "static":
                raise ValueError("static_importance_map requires semantic_budget with static policy")
            if static_importance_map.ndim == 3:
                static_importance_map = static_importance_map.unsqueeze(1)
            if static_importance_map.ndim != 4 or static_importance_map.shape[:2] != (2, 1):
                raise ValueError("static_importance_map must have shape [2,H,W] or [2,1,H,W]")
            self.register_buffer(
                "semantic_static_template", static_importance_map.detach().float().clone()
            )
        else:
            self.semantic_static_template = None
        self.semantic_static_logits = (
            nn.Parameter(torch.zeros(2, 1, 16, 16))
            if fusion_version == "semantic_budget" and spatial_policy == "static" and static_importance_map is None
            else None
        )
        if self.semantic_budget_gate is not None:
            nn.init.zeros_(self.semantic_budget_gate.weight)
            nn.init.zeros_(self.semantic_budget_gate.bias)
        self.last_routing = None
        self.post_fusion_adapter = (
            CATAdapter(
                dim=decoder_dim,
                bottleneck=max(8, int(round(decoder_dim * adapter_ratio))),
            )
            if post_fusion_adapter
            else None
        )
        self.refine = ConvNormAct(decoder_dim, decoder_dim)
        self.prompt_norm = nn.LayerNorm(256) if enable_prompt_fusion else None
        self.prompt_scale = nn.Linear(256, decoder_dim) if enable_prompt_fusion else None
        self.prompt_shift = nn.Linear(256, decoder_dim) if enable_prompt_fusion else None
        self.dense_prompt_proj = (
            nn.Conv2d(256, decoder_dim, 1, bias=False) if enable_spatial_prompt_fusion else None
        )
        self.token_query = (
            nn.Linear(decoder_dim, 64, bias=False) if enable_spatial_prompt_fusion else None
        )
        self.token_key = nn.Linear(256, 64, bias=False) if enable_spatial_prompt_fusion else None
        self.token_value = nn.Linear(256, 64, bias=False) if enable_spatial_prompt_fusion else None
        self.token_output = (
            nn.Linear(64, decoder_dim, bias=False) if enable_spatial_prompt_fusion else None
        )
        self.spatial_prompt_mode = spatial_prompt_mode
        self.classifier = nn.Conv2d(decoder_dim, num_classes, 1)
        if self.prompt_scale is not None and self.prompt_shift is not None:
            nn.init.zeros_(self.prompt_scale.weight)
            nn.init.zeros_(self.prompt_scale.bias)
            nn.init.zeros_(self.prompt_shift.weight)
            nn.init.zeros_(self.prompt_shift.bias)
        if self.dense_prompt_proj is not None:
            nn.init.zeros_(self.dense_prompt_proj.weight)
            nn.init.zeros_(self.token_output.weight)

    def _retained_mask(self, importance: torch.Tensor) -> torch.Tensor:
        """Select the exact top-ratio spatial positions independently per image."""
        if importance.ndim != 4 or importance.shape[1] != 1:
            raise ValueError("importance must have shape [B,1,H,W]")
        flat = importance.flatten(1)
        keep = max(1, min(flat.shape[1], round(flat.shape[1] * self.feature_retention_ratio)))
        mask = torch.zeros_like(flat)
        mask.scatter_(1, flat.topk(keep, dim=1, sorted=False).indices, 1.0)
        return mask.view_as(importance)

    def _spatial_importance(
        self,
        anchor: torch.Tensor,
        value: torch.Tensor,
        evidence: torch.Tensor,
        scale_index: int,
    ) -> torch.Tensor:
        if self.spatial_policy == "adaptive":
            return self.semantic_budget_gate(anchor) + evidence
        if self.spatial_policy == "magnitude":
            return (value - anchor).abs().mean(dim=1, keepdim=True)
        if self.spatial_policy == "random":
            return torch.rand_like(evidence)
        template = (
            self.semantic_static_template
            if self.semantic_static_template is not None
            else self.semantic_static_logits
        )
        return F.interpolate(
            template[scale_index : scale_index + 1],
            size=anchor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).expand(anchor.shape[0], -1, -1, -1)

    def _project_detail(
        self,
        feature: torch.Tensor,
        retained: torch.Tensor,
        layer: nn.Conv2d,
    ) -> tuple[torch.Tensor, bool, torch.Tensor]:
        """Apply a 1x1 lateral projection only at retained spatial positions."""
        if self.feature_retention_ratio >= 1.0 or self.spatial_policy == "magnitude":
            counts = torch.full(
                (feature.shape[0],), feature.shape[-2] * feature.shape[-1],
                dtype=torch.long, device=feature.device,
            )
            return layer(feature), False, counts
        mask = F.interpolate(retained, size=feature.shape[-2:], mode="nearest").bool()
        projected = feature.new_zeros(
            feature.shape[0], layer.out_channels, *feature.shape[-2:]
        )
        weight = layer.weight[:, :, 0, 0]
        counts = mask.flatten(1).sum(1)
        for batch_index in range(feature.shape[0]):
            selected = mask[batch_index, 0]
            if selected.any():
                values = feature[batch_index, :, selected].transpose(0, 1)
                projected_values = F.linear(values, weight, layer.bias)
                projected[batch_index, :, selected] = projected_values.transpose(0, 1)
        return projected, True, counts

    def forward_features(
        self,
        features: dict[str, torch.Tensor],
        prompt_tokens: torch.Tensor | None = None,
        prompt: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        required = set(self.feature_names)
        missing = required - features.keys()
        if missing:
            raise KeyError(f"missing decoder features: {sorted(missing)}")
        if self.fusion_version == "hierarchical":
            start = self.feature_names[-1]
            fused = self.lateral[start](features[start])
            ordered = list(reversed(self.feature_names[:-1]))
            for name in ordered:
                current = self.lateral[name](features[name])
                fused = F.interpolate(fused, size=current.shape[-2:], mode="bilinear", align_corners=False)
                block = self.p3_fuse if name == "P3" else self.p4_fuse
                fused = block(current + fused)
        else:
            if self.fusion_version == "semantic_budget":
                # Embedding is the semantic anchor. P3/P4 are injected only when
                # their content is compatible with the anchor and the image-level
                # budget controller selects the corresponding scale.
                anchor = self.lateral["embedding"](features["embedding"])
                target_size = max(
                    (features[name].shape[-2:] for name in ("P3", "P4", "embedding")),
                    key=lambda s: s[0] * s[1],
                )
                if anchor.shape[-2:] != target_size:
                    anchor = F.interpolate(anchor, size=target_size, mode="bilinear", align_corners=False)
                detail_logits = self.semantic_budget_controller(anchor.mean(dim=(-2, -1)))
                detail_probs = torch.softmax(detail_logits, dim=1)
                k = min(self.representation_budget - 1, 2)
                hard_mask = torch.zeros_like(detail_probs)
                if k:
                    hard_mask.scatter_(1, detail_probs.topk(k, dim=1).indices, 1.0)
                # Straight-through selection: hard routing in the forward pass,
                # soft probabilities in the backward pass.
                route_mask = hard_mask + detail_probs - detail_probs.detach()
                detail_values = []
                detail_effective = []
                importance_maps = []
                retained_masks = []
                sparse_projection_used = []
                projected_positions = []
                student_logits = (
                    self.semantic_budget_student(anchor)
                    if self.semantic_budget_student is not None
                    else None
                )
                teacher_importance_maps = []
                teacher_masks = []
                for index, name in enumerate(("P3", "P4")):
                    selected = self.training or bool(hard_mask[:, index].any())
                    if not selected:
                        detail_values.append(None)
                        detail_effective.append(torch.zeros_like(anchor[:, :1]))
                        importance_maps.append(torch.zeros_like(anchor[:, :1]))
                        retained_masks.append(torch.zeros_like(anchor[:, :1]))
                        sparse_projection_used.append(False)
                        projected_positions.append(
                            torch.zeros(anchor.shape[0], dtype=torch.long, device=anchor.device)
                        )
                        if student_logits is not None:
                            teacher_importance_maps.append(torch.zeros_like(anchor[:, :1]))
                            teacher_masks.append(torch.zeros_like(anchor[:, :1]))
                        continue
                    raw_energy = features[name].float().std(dim=1, keepdim=True)
                    raw_energy = F.interpolate(
                        raw_energy, size=target_size, mode="bilinear", align_corners=False
                    )
                    energy_min = raw_energy.amin(dim=(-2, -1), keepdim=True)
                    energy_max = raw_energy.amax(dim=(-2, -1), keepdim=True)
                    evidence = (raw_energy - energy_min) / (energy_max - energy_min).clamp_min(1e-6)
                    if self.spatial_policy == "magnitude":
                        value = self.lateral[name](features[name])
                        if value.shape[-2:] != target_size:
                            value = F.interpolate(value, size=target_size, mode="bilinear", align_corners=False)
                        importance = self._spatial_importance(anchor, value, evidence, index)
                        sparse_used = False
                        positions = torch.full(
                            (anchor.shape[0],),
                            features[name].shape[-2] * features[name].shape[-1],
                            dtype=torch.long,
                            device=anchor.device,
                        )
                    elif self.spatial_policy == "distilled_magnitude":
                        importance = student_logits[:, index : index + 1]
                        if self.training:
                            value = self.lateral[name](features[name])
                            if value.shape[-2:] != target_size:
                                value = F.interpolate(
                                    value, size=target_size, mode="bilinear", align_corners=False
                                )
                            teacher_importance = (value.detach() - anchor.detach()).abs().mean(
                                dim=1, keepdim=True
                            )
                            teacher_mask = self._retained_mask(teacher_importance)
                            teacher_importance_maps.append(teacher_importance)
                            teacher_masks.append(teacher_mask)
                            sparse_used = False
                            positions = torch.full(
                                (anchor.shape[0],),
                                features[name].shape[-2] * features[name].shape[-1],
                                dtype=torch.long,
                                device=anchor.device,
                            )
                    else:
                        importance = self._spatial_importance(anchor, anchor, evidence, index)
                    retained = self._retained_mask(importance)
                    if self.spatial_policy != "magnitude" and not (
                        self.spatial_policy == "distilled_magnitude" and self.training
                    ):
                        value, sparse_used, positions = self._project_detail(
                            features[name], retained, self.lateral[name]
                        )
                        if value.shape[-2:] != target_size:
                            value = F.interpolate(value, size=target_size, mode="bilinear", align_corners=False)
                    if self.spatial_policy in {"adaptive", "static"}:
                        probability = torch.sigmoid(
                            importance / self.spatial_budget_temperature
                        )
                        retained_st = retained + probability - probability.detach()
                    else:
                        retained_st = retained
                    effective = route_mask[:, index].view(-1, 1, 1, 1) * retained_st
                    detail_values.append(value)
                    detail_effective.append(effective)
                    importance_maps.append(importance)
                    retained_masks.append(retained)
                    sparse_projection_used.append(sparse_used)
                    projected_positions.append(positions)
                fused = anchor
                for index, value in enumerate(detail_values):
                    if value is not None:
                        fused = fused + self.semantic_budget_residual[index] * detail_effective[index] * (value - anchor)
                embedding_weight = torch.ones_like(anchor[:, :1])
                raw_weights = torch.cat([detail_effective[0], detail_effective[1], embedding_weight], dim=1)
                weights = raw_weights / raw_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
                self.last_routing = {
                    "weights": weights.detach(),
                    "entropy": (-(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(1)).detach(),
                    "budget_mask": hard_mask.detach(),
                    "budget": self.representation_budget,
                    "selected_scales": hard_mask.detach().sum(1),
                    "importance_maps": torch.cat(importance_maps, dim=1).detach(),
                    "retained_masks": torch.cat(retained_masks, dim=1).detach(),
                    "retention_ratio": torch.cat(retained_masks, dim=1).mean(dim=(-2, -1)).detach(),
                    "spatial_policy": self.spatial_policy,
                    "target_retention_ratio": self.feature_retention_ratio,
                    "sparse_lateral_projection": any(sparse_projection_used),
                    "lateral_projected_positions": torch.stack(projected_positions, dim=1).detach(),
                }
                if student_logits is not None and self.training:
                    teacher_mask_tensor = torch.cat(teacher_masks, dim=1)
                    student_mask_tensor = torch.cat(retained_masks, dim=1)
                    self.last_routing.update(
                        {
                            "student_logits": student_logits,
                            "teacher_importance_maps": torch.cat(
                                teacher_importance_maps, dim=1
                            ).detach(),
                            "teacher_masks": teacher_mask_tensor.detach(),
                            "teacher_student_agreement": (
                                student_mask_tensor == teacher_mask_tensor
                            ).float().mean().detach(),
                            "teacher_student_mask_iou": (
                                (student_mask_tensor * teacher_mask_tensor).sum()
                                / (
                                    ((student_mask_tensor + teacher_mask_tensor) > 0).sum()
                                ).clamp_min(1)
                            ).detach(),
                        }
                    )
            else:
                aligned = [self.lateral[name](features[name]) for name in self.feature_names]
                target_size = max((x.shape[-2:] for x in aligned), key=lambda s: s[0] * s[1])
                aligned = [F.interpolate(x, size=target_size, mode="bilinear", align_corners=False) if x.shape[-2:] != target_size else x for x in aligned]
                if self.fusion_version == "concat":
                    fused = self.concat_projection(torch.cat(aligned, dim=1))
                elif self.fusion_version == "sum":
                    fused = sum(aligned)
                else:
                    if self.fusion_version == "global":
                        logits = self.global_logits.view(1, -1, 1, 1).expand(aligned[0].shape[0], -1, *target_size)
                    elif self.fusion_version == "image_conditioned":
                        logits = self.image_router(aligned[-1].mean(dim=(-2, -1))).view(aligned[0].shape[0], -1, 1, 1).expand(-1, -1, *target_size)
                    elif self.fusion_version == "scsr":
                        anchor = aligned[-1]
                        scores = []
                        for x in aligned[:2]:
                            scores.append(F.cosine_similarity(anchor, x, dim=1, eps=1e-6))
                        scores.append(self.scsr_confidence(anchor).squeeze(1))
                        logits = torch.stack(scores, dim=1)
                        scales = torch.cat([self.scsr_gamma, torch.ones(1, device=logits.device)])
                        logits = logits * scales.view(1, 3, 1, 1) + self.scsr_beta.view(1, 3, 1, 1)
                    elif self.fusion_version == "scsr_v2":
                        anchor = aligned[-1]
                        anchor_norm = F.normalize(anchor, dim=1, eps=1e-6)
                        scores = []
                        for x in aligned:
                            difference = (x - anchor).abs()
                            compatibility = (
                                F.normalize(x, dim=1, eps=1e-6) * anchor_norm
                            ).sum(dim=1, keepdim=True)
                            detail_magnitude = difference.mean(dim=1, keepdim=True)
                            router_input = torch.cat(
                                (x, anchor, difference, compatibility, detail_magnitude),
                                dim=1,
                            )
                            scores.append(self.scsr_v2_router(router_input))
                        logits = torch.cat(scores, dim=1)
                        temperature = F.softplus(self.scsr_v2_log_temperature) + 0.25
                        logits = (
                            logits + self.scsr_v2_bias.view(1, 3, 1, 1)
                        ) / temperature
                    else:
                        anchor = aligned[-1]
                        anchor_norm = F.normalize(anchor, dim=1, eps=1e-6)
                        scores = []
                        scale_logits = []
                        for index, x in enumerate(aligned):
                            difference = (x - anchor).abs()
                            compatibility = (
                                F.normalize(x, dim=1, eps=1e-6) * anchor_norm
                            ).sum(dim=1, keepdim=True)
                            detail_magnitude = difference.mean(dim=1, keepdim=True)
                            router_input = torch.cat(
                                (x, anchor, difference, compatibility, detail_magnitude),
                                dim=1,
                            )
                            scores.append(self.scsr_v2_router(router_input))
                            scale_logits.append(self.scsr_task_heads[index](x))
                        logits = torch.cat(scores, dim=1)
                        temperature = F.softplus(self.scsr_v2_log_temperature) + 0.25
                        logits = (
                            logits + self.scsr_v2_bias.view(1, 3, 1, 1)
                        ) / temperature
                    soft_weights = torch.softmax(logits, dim=1)
                    weights = soft_weights
                    if self.fusion_version == "scsr_task" and self.task_routing_hard:
                        hard_index = logits.argmax(dim=1)
                        hard_weights = F.one_hot(hard_index, num_classes=3).permute(0, 3, 1, 2).to(logits.dtype)
                        # Forward uses one-hot routing; backward follows the
                        # calibrated soft router through a straight-through estimator.
                        weights = hard_weights + soft_weights - soft_weights.detach()
                    if self.fusion_version in {"scsr_v2", "scsr_task"}:
                        # Residual form keeps the semantic anchor while allowing
                        # P3/P4 to add complementary detail independently.
                        fused = anchor
                        fused = fused + weights[:, 0:1] * (aligned[0] - anchor)
                        fused = fused + weights[:, 1:2] * (aligned[1] - anchor)
                    else:
                        fused = sum(w.unsqueeze(1) * x for w, x in zip(weights.unbind(1), aligned))
                    self.last_routing = {
                        "weights": weights.detach(),
                        "entropy": (-(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(1)).detach(),
                    }
                    if self.fusion_version in {"scsr_v2", "scsr_task"}:
                        self.last_routing["temperature"] = temperature.detach()
                    if self.fusion_version == "scsr_task":
                        self.last_routing["route_weights"] = soft_weights
                        self.last_routing["scale_logits"] = tuple(scale_logits)
        if self.post_fusion_adapter is not None:
            fused = self.post_fusion_adapter(fused)
        fused = self.refine(fused)
        if prompt_tokens is not None:
            if self.prompt_norm is None or self.prompt_scale is None or self.prompt_shift is None:
                raise RuntimeError("decoder prompt fusion is disabled")
            if prompt_tokens.ndim != 3 or prompt_tokens.shape[0] != fused.shape[0]:
                raise ValueError("prompt_tokens must have shape [B,N,256]")
            token_summary = self.prompt_norm(prompt_tokens.mean(dim=1))
            scale = torch.tanh(self.prompt_scale(token_summary)).unsqueeze(-1).unsqueeze(-1)
            shift = self.prompt_shift(token_summary).unsqueeze(-1).unsqueeze(-1)
            fused = fused * (1.0 + scale) + shift
        if prompt is not None:
            if any(module is None for module in (self.dense_prompt_proj, self.token_query, self.token_key, self.token_value, self.token_output)):
                raise RuntimeError("decoder spatial prompt fusion is disabled")
            dense_prompt = prompt.get("dense_prompt")
            token_prompt = prompt.get("token_prompt")
            if self.spatial_prompt_mode in {"both", "dense"}:
                if dense_prompt is None:
                    raise KeyError("spatial prompt requires dense_prompt")
                if dense_prompt.ndim != 4 or dense_prompt.shape[:2] != (fused.shape[0], 256):
                    raise ValueError("dense_prompt must have shape [B,256,H,W]")
                dense = F.interpolate(dense_prompt, fused.shape[-2:], mode="bilinear", align_corners=False)
                fused = fused + self.dense_prompt_proj(dense)
            if self.spatial_prompt_mode in {"both", "token"}:
                if token_prompt is None:
                    raise KeyError("spatial prompt requires token_prompt")
                if token_prompt.ndim != 3 or token_prompt.shape[0] != fused.shape[0] or token_prompt.shape[2] != 256:
                    raise ValueError("token_prompt must have shape [B,N,256]")
                query = self.token_query(fused.flatten(2).transpose(1, 2))
                key = self.token_key(token_prompt)
                value = self.token_value(token_prompt)
                attention = torch.softmax(query @ key.transpose(-2, -1) / 8.0, dim=-1)
                modulation = self.token_output(attention @ value).transpose(1, 2).reshape_as(fused)
                fused = fused + modulation
        return fused

    def forward(
        self,
        features: dict[str, torch.Tensor],
        output_size: tuple[int, int],
        prompt_tokens: torch.Tensor | None = None,
        prompt: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        fused = self.forward_features(features, prompt_tokens=prompt_tokens, prompt=prompt)
        logits = self.classifier(fused)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
