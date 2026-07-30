"""Train prototype-conditioned semantic queries on a fixed NEU_Seg K-shot subset."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adasam.adapters import CATAdapter  # noqa: E402
from adasam.backbone import MultiScaleMobileSAMBackbone, build_mobile_sam  # noqa: E402
from adasam.datasets import NEUSegDataset  # noqa: E402
from adasam.losses import PrototypeQuerySemanticLoss  # noqa: E402
from adasam.model import (  # noqa: E402
    PrototypeConditionedSemanticQueryDecoder,
    PrototypeQueryConfig,
)
from adasam.utils import set_seed  # noqa: E402
from adasam.utils.transforms import preprocess_image, resize_mask  # noqa: E402

FOREGROUND_CLASSES = (1, 2, 3)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _REPO_ROOT / path


def load_stage1(path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("stage") != "neuseg_stage1":
        raise ValueError(f"not a NEU Stage 1 checkpoint: {path}")
    if "adapter" not in checkpoint or "kshot_manifest" not in checkpoint:
        raise ValueError("Stage 1 checkpoint lacks adapter or kshot_manifest")
    return checkpoint


def manifest_indices(dataset: NEUSegDataset, manifest: dict) -> list[int]:
    by_name = {name: idx for idx, name in enumerate(dataset.sample_names)}
    missing = [name for name in manifest["sample_ids"] if name not in by_name]
    if missing:
        raise ValueError(f"manifest samples not found: {missing}")
    return [by_name[name] for name in manifest["sample_ids"]]


class Stage2Trainer:
    def __init__(self, cfg: dict, args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.args = args
        self.seed = int(cfg.get("seed", 42))
        set_seed(self.seed)
        requested = cfg["train"].get("device", "cuda")
        self.device = torch.device(requested if torch.cuda.is_available() else "cpu")
        self.stage1_path = resolve_path(args.stage1_ckpt)
        self.stage1 = load_stage1(self.stage1_path, self.device)
        self.manifest = self.stage1["kshot_manifest"]
        self.k_shot = int(self.manifest["k_shot"])

        data_root = resolve_path(cfg["data"]["data_root"])
        self.train_ds = NEUSegDataset(data_root, split="train")
        self.val_ds = NEUSegDataset(data_root, split="test")
        self.selected_indices = manifest_indices(self.train_ds, self.manifest)
        self.class_indices = self._class_indices()
        requested_support = int(cfg["fewshot"].get("support_shot", 1))
        self.support_shot = min(requested_support, self.k_shot)
        if self.k_shot < 2:
            print("[NEU Stage2] WARNING: K=1 reuses one image as support and query")

        backbone_cfg = cfg["backbone"]
        sam = build_mobile_sam(
            resolve_path(backbone_cfg["checkpoint"]),
            backbone_cfg.get("model_type", "vit_t"),
            self.device,
        )
        self.backbone = MultiScaleMobileSAMBackbone(
            sam.image_encoder, sam.image_encoder.img_size
        ).to(self.device)
        adapter_cfg = self.stage1.get("config", {}).get("adapter", {})
        self.adapter = CATAdapter(
            dim=256, bottleneck=int(adapter_cfg.get("bottleneck", 64))
        ).to(self.device)
        self.adapter.load_state_dict(self.stage1["adapter"])
        self.adapter.eval()
        for parameter in self.adapter.parameters():
            parameter.requires_grad_(False)

        query_cfg = PrototypeQueryConfig.from_dict(cfg.get("semantic_query", {}))
        self.model = PrototypeConditionedSemanticQueryDecoder(query_cfg).to(self.device)
        loss_cfg = dict(cfg.get("semantic_query_loss", {}))
        loss_cfg.pop("class_loss_weights", None)
        self.criterion = PrototypeQuerySemanticLoss(**loss_cfg)
        train_cfg = cfg["train"]
        self.epochs = int(train_cfg.get("epochs", 100))
        self.episodes = int(train_cfg.get("episodes_per_epoch", 100))
        self.val_every = int(train_cfg.get("val_every", 0))
        self.val_samples = int(train_cfg.get("val_samples", 120))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.class_thresholds = list(cfg.get("eval", {}).get("class_thresholds", []))
        self.calibrate_thresholds = not bool(self.class_thresholds)
        self.decoder_lr = float(train_cfg.get("lr", 1e-4))
        self.adapter_lr = float(train_cfg.get("adapter_lr", self.decoder_lr / 10.0))
        self.backbone_lr = float(train_cfg.get("backbone_lr", self.decoder_lr / 20.0))
        self.unfreeze_after = int(train_cfg.get("unfreeze_after", 10**9))
        self.optimizer = self._build_optimizer()
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(self.epochs, 1))
        self.rng = random.Random(self.seed)
        output_root = resolve_path(cfg.get("output_dir", "runs"))
        self.out_dir = output_root / f"neuseg_semantic_query_stage2_k{self.k_shot}_seed{self.seed}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[NEU Query Stage2] device={self.device} k={self.k_shot} "
            f"train_support<=K-1 eval_support=all-K({self.k_shot}) queries={query_cfg.num_queries}"
        )
        print(f"[NEU Query Stage2] output={self.out_dir}")

    def _class_indices(self) -> dict[int, list[int]]:
        result = {class_id: [] for class_id in FOREGROUND_CLASSES}
        for idx in self.selected_indices:
            labels = torch.unique(self.train_ds[idx]["masks"])
            for class_id in FOREGROUND_CLASSES:
                if (labels == class_id).any():
                    result[class_id].append(idx)
        for class_id, indices in result.items():
            if not indices:
                raise ValueError(f"no {self.train_ds.CLASS_NAMES[class_id]} sample in manifest")
        return result

    def _build_optimizer(self) -> AdamW:
        groups = [{"params": self.model.parameters(), "lr": self.decoder_lr}]
        if any(parameter.requires_grad for parameter in self.adapter.parameters()):
            groups.append({"params": self.adapter.parameters(), "lr": self.adapter_lr})
        if any(parameter.requires_grad for parameter in self.backbone.parameters()):
            groups.append({"params": [p for p in self.backbone.parameters() if p.requires_grad], "lr": self.backbone_lr})
        return AdamW(groups, weight_decay=float(self.cfg["train"].get("weight_decay", 1e-4)))

    def _embed(self, image: torch.Tensor) -> tuple[dict[str, torch.Tensor], object]:
        processed, meta = preprocess_image(image)
        features = self.backbone(processed.unsqueeze(0).to(self.device))
        features["stage3"] = self.adapter(features["stage3"])
        return features, meta

    def _augment(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rng.random() < 0.5:
            image, mask = image.flip(-1), mask.flip(-1)
        if self.rng.random() < 0.5:
            image, mask = image.flip(-2), mask.flip(-2)
        angle = self.rng.uniform(-12.0, 12.0) * math.pi / 180.0
        scale = self.rng.uniform(0.75, 1.25)
        theta = image.new_tensor(
            [[scale * math.cos(angle), -scale * math.sin(angle), 0.0],
             [scale * math.sin(angle), scale * math.cos(angle), 0.0]]
        )[None]
        grid = F.affine_grid(theta, (1, *image.shape), align_corners=False)
        image = F.grid_sample(image[None], grid, mode="bilinear", padding_mode="reflection", align_corners=False)[0]
        mask = F.grid_sample(mask[None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)[0].to(mask.dtype)
        brightness = self.rng.uniform(0.8, 1.2)
        contrast = self.rng.uniform(0.8, 1.2)
        image = ((image - image.mean(dim=(-2, -1), keepdim=True)) * contrast + image.mean(dim=(-2, -1), keepdim=True)) * brightness
        image = image.clamp(0.0, 1.0)
        image = image.clamp_min(1e-4).pow(self.rng.uniform(0.8, 1.2))
        if self.rng.random() < 0.35:
            image = (image + torch.randn_like(image) * self.rng.uniform(0.005, 0.03)).clamp(0.0, 1.0)
        if self.rng.random() < 0.25:
            image = F.avg_pool2d(image[None], 3, stride=1, padding=1)[0]
        return image, mask

    def _copy_paste(self, image: torch.Tensor, mask: torch.Tensor, class_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rng.random() >= float(self.cfg["fewshot"].get("copy_paste_prob", 0.5)):
            return image, mask
        donors = self.class_indices[class_id]
        donor = self.train_ds[self.rng.choice(donors)]
        donor_mask = donor["masks"].squeeze(0) == class_id
        if not donor_mask.any() or donor["image"].shape[-2:] != image.shape[-2:]:
            return image, mask
        pasted = image.clone()
        pasted[:, donor_mask] = donor["image"][:, donor_mask]
        mask = mask.clone()
        mask[:, donor_mask] = class_id
        return pasted, mask

    def _support(
        self, indices: list[int], class_id: int, augment: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, masks = [], []
        first_image = self.train_ds[indices[0]]["image"]
        if augment:
            first_image, _ = self._augment(first_image, self.train_ds[indices[0]]["masks"])
        grid_size = self._embed(first_image)[0]["stage3"].shape[-2:]
        for idx in indices:
            image, raw_mask = self.train_ds[idx]["image"], self.train_ds[idx]["masks"]
            if augment:
                image, raw_mask = self._augment(image, raw_mask)
            features.append(self._embed(image)[0]["stage3"][0])
            masks.append(resize_mask(raw_mask.squeeze(0) == class_id, grid_size))
        return torch.stack(features), torch.stack(masks).to(self.device)

    def _sample_episode(self) -> tuple[int, list[int], int]:
        weights = self.cfg["fewshot"].get("class_sampling_weights", [0.45, 0.25, 0.30])
        class_id = self.rng.choices(FOREGROUND_CLASSES, weights=weights, k=1)[0]
        negative = self.rng.random() < float(self.cfg.get("fewshot", {}).get("negative_episode_prob", 0.4))
        candidates = self.class_indices[class_id]
        absent = [idx for idx in self.selected_indices if idx not in candidates]
        query_idx = self.rng.choice(absent if negative and absent else candidates)
        pool = [idx for idx in candidates if idx != query_idx] or [query_idx]
        support = self.rng.sample(pool, min(self.support_shot, len(pool)))
        return class_id, support, query_idx

    def train_episode(self) -> dict[str, float]:
        class_id, support_indices, query_idx = self._sample_episode()
        support_features, support_masks = self._support(support_indices, class_id, augment=True)
        query_image, query_mask = self._copy_paste(
            self.train_ds[query_idx]["image"], self.train_ds[query_idx]["masks"], class_id
        )
        query_image, query_mask = self._augment(query_image, query_mask)
        query_features = self._embed(query_image)[0]
        semantic_target = query_mask.squeeze(0) == class_id
        output = self.model(query_features, support_features, support_masks)
        if self.criterion.consistency_weight > 0:
            switched_class = self.rng.choice([c for c in FOREGROUND_CLASSES if c != class_id])
            switched_indices = self.class_indices[switched_class][: self.support_shot]
            switched_support = self._support(switched_indices, switched_class, augment=True)
            switched_output = self.model(query_features, *switched_support)
            switched_target = query_mask.squeeze(0) == switched_class
        class_weights = self.cfg["semantic_query_loss"].get("class_loss_weights", [1.5, 1.0, 1.15])
        losses = self.criterion(
            output, semantic_target[None].to(self.device), class_weight=float(class_weights[class_id - 1])
        )
        if self.criterion.consistency_weight > 0:
            switched_losses = self.criterion(
                switched_output, switched_target[None].to(self.device), class_weight=float(class_weights[switched_class - 1])
            )
            losses["consistency"] = switched_losses["loss"]
            losses["loss"] = losses["loss"] + self.criterion.consistency_weight * switched_losses["loss"]
        self.optimizer.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return {key: float(value.detach()) for key, value in losses.items()}

    @torch.no_grad()
    def support_cache(self) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        return {
            class_id: self._support(indices, class_id, augment=False)
            for class_id, indices in self.class_indices.items()
        }

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        supports = self.support_cache()
        cached_predictions: list[tuple[torch.Tensor, torch.Tensor]] = []
        if self.val_samples >= len(self.val_ds):
            val_indices = list(range(len(self.val_ds)))
        else:
            pools = {
                class_id: [
                    i for i in range(len(self.val_ds))
                    if (self.val_ds[i]["masks"] == class_id).any()
                ]
                for class_id in FOREGROUND_CLASSES
            }
            val_indices, selected = [], set()
            cursor = {class_id: 0 for class_id in FOREGROUND_CLASSES}
            while len(val_indices) < min(self.val_samples, len(self.val_ds)):
                added = False
                for class_id in FOREGROUND_CLASSES:
                    pool = pools[class_id]
                    while cursor[class_id] < len(pool) and pool[cursor[class_id]] in selected:
                        cursor[class_id] += 1
                    if cursor[class_id] < len(pool):
                        idx = pool[cursor[class_id]]
                        cursor[class_id] += 1
                        selected.add(idx)
                        val_indices.append(idx)
                        added = True
                        if len(val_indices) >= self.val_samples:
                            break
                if not added:
                    remaining = [i for i in range(len(self.val_ds)) if i not in selected]
                    val_indices.extend(remaining[: self.val_samples - len(val_indices)])
                    break
        for idx in tqdm(val_indices, desc="validate", leave=False):
            sample = self.val_ds[idx]
            query_features, _ = self._embed(sample["image"])
            class_probabilities = []
            for class_id in FOREGROUND_CLASSES:
                output = self.model(query_features, *supports[class_id])
                probability = self.model.semantic_probability(output)
                probability = F.interpolate(
                    probability[:, None], sample["image_size"], mode="bilinear", align_corners=False
                )[0, 0]
                class_probabilities.append(probability.cpu())
            cached_predictions.append((torch.stack(class_probabilities), sample["masks"].squeeze(0).cpu()))
        if self.calibrate_thresholds:
            thresholds = []
            for class_id in FOREGROUND_CLASSES:
                best_iou, best_threshold = -1.0, 0.5
                for threshold in torch.linspace(0.1, 0.9, 17):
                    inter, union = 0.0, 0.0
                    for probabilities, target in cached_predictions:
                        prediction = probabilities[class_id - 1] >= threshold
                        truth = target == class_id
                        inter += (prediction & truth).sum().item()
                        union += (prediction | truth).sum().item()
                    score = inter / max(union, 1.0)
                    if score > best_iou:
                        best_iou, best_threshold = score, float(threshold)
                thresholds.append(best_threshold)
            self.class_thresholds = thresholds
        inter = torch.zeros(4, dtype=torch.float64)
        union = torch.zeros(4, dtype=torch.float64)
        thresholds = torch.tensor(self.class_thresholds)
        for probabilities, target in cached_predictions:
            adjusted = probabilities / thresholds[:, None, None]
            confidence, class_index = adjusted.max(dim=0)
            prediction = class_index + 1
            prediction[confidence < 1.0] = 0
            for class_id in range(4):
                pred_class, gt_class = prediction == class_id, target == class_id
                inter[class_id] += (pred_class & gt_class).sum().item()
                union[class_id] += (pred_class | gt_class).sum().item()
        ious = [float(inter[c] / union[c]) if union[c] else None for c in range(4)]
        foreground = [value for value in ious[1:] if value is not None]
        self.model.train()
        return {
            "mIoU_fg": sum(foreground) / max(len(foreground), 1),
            "per_class_iou": {self.val_ds.CLASS_NAMES[c]: ious[c] for c in range(4)},
            "n_samples": len(val_indices),
            "class_thresholds": {
                self.val_ds.CLASS_NAMES[class_id]: self.class_thresholds[class_id - 1]
                for class_id in FOREGROUND_CLASSES
            },
        }

    def save(self, path: Path, epoch: int, metrics: dict) -> None:
        torch.save(
            {
                "stage": "neuseg_prototype_semantic_query_stage2",
                "epoch": epoch,
                "model": self.model.state_dict(),
                "adapter": self.adapter.state_dict(),
                "config": self.cfg,
                "kshot_manifest": self.manifest,
                "stage1_checkpoint": str(self.stage1_path),
                "metrics": metrics,
                "class_thresholds": self.class_thresholds,
            },
            path,
        )

    def train(self) -> Path:
        best_path = self.out_dir / "best_model.pt"
        best_score = -1.0
        max_episodes = self.args.steps or self.episodes
        for epoch in range(1, self.epochs + 1):
            if epoch == self.unfreeze_after:
                for parameter in self.adapter.parameters():
                    parameter.requires_grad_(True)
                self.backbone.unfreeze_last_stage()
                self.optimizer = self._build_optimizer()
                print(f"[NEU Query Stage2] unfroze CATAdapter and TinyViT last stage at epoch={epoch}")
            self.model.train()
            self.backbone.train()
            self.adapter.train()
            epoch_metrics = []
            for _ in tqdm(range(max_episodes), desc=f"epoch {epoch}/{self.epochs}"):
                epoch_metrics.append(self.train_episode())
            self.scheduler.step()
            means = {
                key: sum(item[key] for item in epoch_metrics) / len(epoch_metrics)
                for key in epoch_metrics[0]
            }
            print(
                f"[NEU Query Stage2] epoch={epoch} loss={means['loss']:.4f} "
                f"bce={means['bce']:.4f} dice={means['dice']:.4f} "
                f"diversity={means['diversity']:.4f}"
            )
            record: dict = {"epoch": epoch, **means}
            if self.val_every > 0 and (epoch % self.val_every == 0 or epoch == self.epochs):
                validation = self.validate()
                record["validation"] = validation
                score = validation["mIoU_fg"]
                print(f"[NEU Query Stage2] validation mIoU_fg={score:.4f}")
                if score > best_score:
                    best_score = score
                    self.save(best_path, epoch, {"train": means, "validation": validation})
            elif not best_path.exists():
                self.save(best_path, epoch, {"train": means})
            with (self.out_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        self.save(self.out_dir / "last_model.pt", self.epochs, {"train": means})
        return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NEU prototype-conditioned semantic-query Stage 2")
    parser.add_argument("--config", default=str(_REPO_ROOT / "configs/neu_seg_stage2.yaml"))
    parser.add_argument("--stage1-ckpt", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--steps", type=int, help="Maximum episodes per epoch")
    parser.add_argument("--support-shot", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--val-samples", type=int)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    overrides = [
        (("train", "epochs"), args.epochs),
        (("train", "episodes_per_epoch"), args.episodes),
        (("train", "device"), args.device),
        (("train", "val_samples"), args.val_samples),
        (("fewshot", "support_shot"), args.support_shot),
        (("data", "data_root"), args.data_root),
        (("seed",), args.seed),
        (("output_dir",), args.output_dir),
    ]
    for keys, value in overrides:
        if value is None:
            continue
        target = cfg
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = value
    return cfg


def main() -> None:
    args = parse_args()
    best = Stage2Trainer(load_config(args), args).train()
    print(f"[NEU Query Stage2] done: {best}")


if __name__ == "__main__":
    main()
