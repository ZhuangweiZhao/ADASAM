"""NEU_Seg Stage 1: limited-data domain adaptation.

Select K labelled training images for each foreground class, then train only a
CATAdapter and an auxiliary four-class segmentation head on top of a frozen
MobileSAM encoder. The auxiliary head is not consumed by Stage 2.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

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

from adasam.adapters import CATAdapter  # noqa: E402
from adasam.backbone import MobileSAMBackbone, build_mobile_sam  # noqa: E402
from adasam.datasets import NEUSegDataset  # noqa: E402
from adasam.utils import set_seed  # noqa: E402
from adasam.utils.transforms import preprocess_image, resize_mask  # noqa: E402


class SegHead(nn.Module):
    def __init__(self, in_dim: int = 256, num_classes: int = 4) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_dim, num_classes, 1)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(self.head(x), size=size, mode="bilinear", align_corners=False)


def multiclass_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=1)
    one_hot = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    inter = (probs * one_hot).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    return (1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)).mean()


def hard_negative_correlation_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 0.1,
    negative_ratio: float = 1.0,
) -> torch.Tensor:
    """Separate foreground correlations from the hardest background pixels."""
    resized_targets = F.interpolate(
        targets[:, None].float(), size=features.shape[-2:], mode="nearest"
    )[:, 0].long()
    normalized_features = F.normalize(features, dim=1)
    losses = []
    for batch_index in range(features.shape[0]):
        feature = features[batch_index]
        normalized = normalized_features[batch_index]
        labels = resized_targets[batch_index]
        for class_id in (1, 2, 3):
            positive_mask = labels == class_id
            negative_mask = labels != class_id
            positive_count = int(positive_mask.sum().item())
            negative_count = int(negative_mask.sum().item())
            if positive_count == 0 or negative_count == 0:
                continue
            prototype = F.normalize(feature[:, positive_mask].mean(dim=1), dim=0)
            correlation = torch.einsum("chw,c->hw", normalized, prototype)
            positive = correlation[positive_mask].mean()
            hard_k = max(1, min(negative_count, round(positive_count * negative_ratio)))
            hard_negative = correlation[negative_mask].topk(hard_k).values.mean()
            losses.append(F.relu(features.new_tensor(margin) - (positive - hard_negative)))
    return torch.stack(losses).mean() if losses else features.sum() * 0.0


def select_k_shot(dataset: NEUSegDataset, k_shot: int, seed: int) -> list[int]:
    """Select a deterministic union containing at least K images per FG class."""
    if k_shot < 1:
        raise ValueError("k_shot must be >= 1")
    candidates: dict[int, list[int]] = {1: [], 2: [], 3: []}
    for idx in range(len(dataset)):
        labels = torch.unique(dataset[idx]["masks"])
        for class_id in candidates:
            if (labels == class_id).any():
                candidates[class_id].append(idx)

    rng = random.Random(seed)
    selected: set[int] = set()
    for class_id, indices in candidates.items():
        if len(indices) < k_shot:
            raise ValueError(
                f"class {class_id} ({dataset.CLASS_NAMES[class_id]}) has only "
                f"{len(indices)} images, requested k_shot={k_shot}"
            )
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        selected.update(shuffled[:k_shot])
    return sorted(selected)


class Trainer:
    def __init__(self, cfg: dict, args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.args = args
        self.seed = int(cfg.get("seed", 42))
        set_seed(self.seed)
        requested_device = cfg["train"].get("device", "cuda")
        self.device = torch.device(requested_device if torch.cuda.is_available() else "cpu")

        data_root = Path(cfg["data"]["data_root"])
        if not data_root.is_absolute():
            data_root = _REPO_ROOT / data_root
        self.train_ds = NEUSegDataset(data_root, split="train")
        self.val_ds = NEUSegDataset(data_root, split="test")
        self.k_shot = int(cfg["fewshot"]["k_shot"])
        self.train_indices = select_k_shot(self.train_ds, self.k_shot, self.seed)

        tcfg = cfg["train"]
        self.epochs = int(tcfg.get("epochs", 100))
        self.batch_size = int(tcfg.get("batch_size", 2))
        self.val_every = int(tcfg.get("val_every", 5))
        self.val_samples = int(tcfg.get("val_samples", 120))
        self.max_steps = args.steps

        checkpoint = Path(cfg["backbone"]["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = _REPO_ROOT / checkpoint
        sam = build_mobile_sam(
            checkpoint, cfg["backbone"].get("model_type", "vit_t"), self.device
        )
        self.backbone = MobileSAMBackbone(
            sam.image_encoder, sam.image_encoder.img_size
        ).to(self.device)
        self.adapter = CATAdapter(
            dim=256, bottleneck=int(cfg.get("adapter", {}).get("bottleneck", 64))
        ).to(self.device)
        self.seg_head = SegHead(num_classes=4).to(self.device)

        params = list(self.adapter.parameters()) + list(self.seg_head.parameters())
        self.optimizer = AdamW(
            params,
            lr=float(tcfg.get("lr", 1e-3)),
            weight_decay=float(tcfg.get("weight_decay", 1e-4)),
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(self.epochs, 1))
        self.grad_clip = float(tcfg.get("grad_clip", 1.0))
        self.ce_weight = float(cfg.get("loss", {}).get("ce_weight", 1.0))
        self.dice_weight = float(cfg.get("loss", {}).get("dice_weight", 1.0))
        self.correlation_loss_weight = float(
            cfg.get("loss", {}).get("correlation_margin_weight", 0.2)
        )
        self.correlation_loss_margin = float(
            cfg.get("loss", {}).get("correlation_margin", 0.1)
        )
        self.hard_negative_ratio = float(
            cfg.get("loss", {}).get("hard_negative_ratio", 1.0)
        )
        selection_cfg = cfg.get("domain_selection", {})
        self.retrieval_weight = float(selection_cfg.get("retrieval_weight", 0.4))
        self.correlation_margin_weight = float(
            selection_cfg.get("correlation_margin_weight", 0.3)
        )
        self.localization_recall_weight = float(
            selection_cfg.get("localization_recall_weight", 0.3)
        )
        self.localization_topk = int(selection_cfg.get("localization_topk", 20))
        self.checkpoint_every = int(tcfg.get("checkpoint_every", 10))

        output_root = Path(cfg.get("output_dir", "runs"))
        if not output_root.is_absolute():
            output_root = _REPO_ROOT / output_root
        self.out_dir = output_root / f"neuseg_stage1_k{self.k_shot}_seed{self.seed}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = {
            "k_shot": self.k_shot,
            "seed": self.seed,
            "sample_ids": [self.train_ds.sample_names[i] for i in self.train_indices],
        }
        (self.out_dir / "kshot_manifest.json").write_text(
            json.dumps(self.manifest, indent=2), encoding="utf-8"
        )
        print(
            f"[NEU Stage1] device={self.device} k={self.k_shot} "
            f"selected={len(self.train_indices)} val={min(len(self.val_ds), self.val_samples)}"
        )
        print(f"[NEU Stage1] samples={self.manifest['sample_ids']}")
        print(f"[NEU Stage1] output={self.out_dir}")

    def _batch(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        images, targets = [], []
        for idx in indices:
            sample = self.train_ds[idx]
            image, _ = preprocess_image(sample["image"])
            images.append(image)
            targets.append(sample["masks"].squeeze(0).long())
        return torch.stack(images).to(self.device), torch.stack(targets).to(self.device)

    @torch.no_grad()
    def validate_domain_adaptation(self) -> dict:
        """Measure prototype separability and localization, independent of SegHead."""
        self.adapter.eval()
        self.seg_head.eval()

        support_vectors: dict[int, list[torch.Tensor]] = {1: [], 2: [], 3: []}
        for idx in self.train_indices:
            sample = self.train_ds[idx]
            image, _ = preprocess_image(sample["image"])
            embedding = self.backbone(image.unsqueeze(0).to(self.device))["image_embedding"]
            feature = self.adapter(embedding)[0]
            grid_size = feature.shape[-2:]
            labels = torch.unique(sample["masks"])
            for class_id in (1, 2, 3):
                if not (labels == class_id).any():
                    continue
                mask = resize_mask(sample["masks"].squeeze(0) == class_id, grid_size).to(self.device)
                prototype = (feature * mask[None]).sum(dim=(1, 2)) / mask.sum().clamp_min(1.0)
                support_vectors[class_id].append(prototype)
        class_prototypes = {
            class_id: F.normalize(torch.stack(vectors).mean(0), dim=0)
            for class_id, vectors in support_vectors.items()
            if vectors
        }

        retrieval_correct = 0
        retrieval_total = 0
        correlation_margins = []
        localization_hits = []
        limit = min(len(self.val_ds), self.val_samples)
        for idx in range(limit):
            sample = self.val_ds[idx]
            image, _ = preprocess_image(sample["image"])
            embedding = self.backbone(image.unsqueeze(0).to(self.device))["image_embedding"]
            feature = self.adapter(embedding)[0]
            grid_size = feature.shape[-2:]
            feature_normalized = F.normalize(feature, dim=0)
            labels = torch.unique(sample["masks"])
            for class_id in (1, 2, 3):
                if class_id not in class_prototypes or not (labels == class_id).any():
                    continue
                mask = resize_mask(sample["masks"].squeeze(0) == class_id, grid_size).to(self.device)
                if mask.sum() < 1:
                    continue
                query_prototype = (feature * mask[None]).sum(dim=(1, 2)) / mask.sum()
                query_prototype = F.normalize(query_prototype, dim=0)
                similarities = {
                    candidate: torch.dot(query_prototype, prototype).item()
                    for candidate, prototype in class_prototypes.items()
                }
                prediction = max(similarities, key=similarities.get)
                retrieval_correct += int(prediction == class_id)
                retrieval_total += 1

                correlation = torch.einsum(
                    "chw,c->hw", feature_normalized, class_prototypes[class_id]
                )
                positive = correlation[mask > 0.5].mean()
                negative = correlation[mask <= 0.5].mean()
                correlation_margins.append(float((positive - negative).item()))
                k = min(self.localization_topk, correlation.numel())
                top_indices = correlation.flatten().topk(k).indices
                hit = mask.flatten()[top_indices].max().item() > 0.5
                localization_hits.append(float(hit))

        retrieval = retrieval_correct / max(retrieval_total, 1)
        margin = sum(correlation_margins) / max(len(correlation_margins), 1)
        recall = sum(localization_hits) / max(len(localization_hits), 1)
        normalized_margin = max(0.0, min(1.0, (margin + 1.0) / 2.0))
        domain_score = (
            self.retrieval_weight * retrieval
            + self.correlation_margin_weight * normalized_margin
            + self.localization_recall_weight * recall
        )
        return {
            "domain_score": domain_score,
            "prototype_retrieval_at_1": retrieval,
            "correlation_margin": margin,
            f"localization_recall_at_{self.localization_topk}": recall,
            "n_class_queries": retrieval_total,
        }

    def save(self, path: Path, epoch: int, metrics: dict) -> None:
        torch.save(
            {
                "stage": "neuseg_stage1",
                "epoch": epoch,
                "adapter": self.adapter.state_dict(),
                "seg_head": self.seg_head.state_dict(),
                "config": self.cfg,
                "kshot_manifest": self.manifest,
                "metrics": metrics,
            },
            path,
        )

    def train(self) -> Path:
        rng = random.Random(self.seed)
        best_selection = (-1.0, float("-inf"))
        best_domain_score = -1.0
        best_margin = float("-inf")
        best_path = self.out_dir / "best_adapter.pt"
        metrics_path = self.out_dir / "metrics.jsonl"
        metrics_path.write_text("", encoding="utf-8")
        params = list(self.adapter.parameters()) + list(self.seg_head.parameters())
        for epoch in range(1, self.epochs + 1):
            order = self.train_indices.copy()
            rng.shuffle(order)
            losses, correlation_losses = [], []
            self.adapter.train()
            self.seg_head.train()
            batches = [order[i : i + self.batch_size] for i in range(0, len(order), self.batch_size)]
            if self.max_steps is not None:
                batches = batches[: self.max_steps]
            for batch_indices in tqdm(batches, desc=f"epoch {epoch}/{self.epochs}"):
                images, targets = self._batch(batch_indices)
                with torch.no_grad():
                    embedding = self.backbone(images)["image_embedding"]
                features = self.adapter(embedding)
                logits = self.seg_head(features, tuple(targets.shape[-2:]))
                ce = F.cross_entropy(logits, targets)
                dice = multiclass_dice_loss(logits, targets)
                correlation_loss = hard_negative_correlation_loss(
                    features,
                    targets,
                    margin=self.correlation_loss_margin,
                    negative_ratio=self.hard_negative_ratio,
                )
                loss = (
                    self.ce_weight * ce
                    + self.dice_weight * dice
                    + self.correlation_loss_weight * correlation_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, self.grad_clip)
                self.optimizer.step()
                losses.append(float(loss.item()))
                correlation_losses.append(float(correlation_loss.item()))
            self.scheduler.step()
            avg_loss = sum(losses) / max(len(losses), 1)
            avg_correlation_loss = sum(correlation_losses) / max(len(correlation_losses), 1)
            should_validate = self.val_every > 0 and (
                epoch % self.val_every == 0 or epoch == self.epochs
            )
            if should_validate:
                metrics = self.validate_domain_adaptation()
                score = metrics["domain_score"]
                localization = metrics[f"localization_recall_at_{self.localization_topk}"]
                margin = metrics["correlation_margin"]
                checkpoint_metrics = {
                    **metrics,
                    "train_loss": avg_loss,
                    "train_correlation_loss": avg_correlation_loss,
                }
                print(
                    f"[NEU Stage1] epoch={epoch} loss={avg_loss:.4f} "
                    f"corr_loss={avg_correlation_loss:.4f} "
                    f"domain_score={score:.4f} "
                    f"retrieval@1={metrics['prototype_retrieval_at_1']:.4f} "
                    f"corr_margin={metrics['correlation_margin']:.4f} "
                    f"loc_recall@{self.localization_topk}="
                    f"{metrics[f'localization_recall_at_{self.localization_topk}']:.4f}"
                )
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"epoch": epoch, **checkpoint_metrics}) + "\n")
                selection = (localization, margin)
                if selection > best_selection:
                    best_selection = selection
                    self.save(best_path, epoch, checkpoint_metrics)
                if score > best_domain_score:
                    best_domain_score = score
                    self.save(self.out_dir / "best_domain_adapter.pt", epoch, checkpoint_metrics)
                if margin > best_margin:
                    best_margin = margin
                    self.save(self.out_dir / "best_margin_adapter.pt", epoch, checkpoint_metrics)
                if self.checkpoint_every > 0 and epoch % self.checkpoint_every == 0:
                    self.save(self.out_dir / f"adapter_epoch_{epoch:03d}.pt", epoch, checkpoint_metrics)
            else:
                print(
                    f"[NEU Stage1] epoch={epoch} loss={avg_loss:.4f} "
                    f"corr_loss={avg_correlation_loss:.4f}"
                )
                if avg_loss < 0 or not best_path.exists():
                    self.save(best_path, epoch, {"train_loss": avg_loss})
        self.save(self.out_dir / "last_adapter.pt", self.epochs, {"train_loss": avg_loss})
        return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NEU_Seg limited-data Stage 1")
    parser.add_argument("--config", default=str(_REPO_ROOT / "configs/neu_seg_stage1.yaml"))
    parser.add_argument("--k-shot", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps", type=int, help="Maximum batches per epoch")
    parser.add_argument("--batch-size", type=int)
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
        (("fewshot", "k_shot"), args.k_shot),
        (("train", "epochs"), args.epochs),
        (("train", "batch_size"), args.batch_size),
        (("train", "device"), args.device),
        (("train", "val_samples"), args.val_samples),
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
    trainer = Trainer(load_config(args), args)
    best = trainer.train()
    print(f"[NEU Stage1] done: {best}")


if __name__ == "__main__":
    main()
