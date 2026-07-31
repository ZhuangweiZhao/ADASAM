"""Full-supervision U-Net baseline for the official NEU_Seg split."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
UNET_ROOT = REPO_ROOT / "thirdparty" / "Pytorch-UNet"
for path in (REPO_ROOT, UNET_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from adasam.datasets import NEUSegDataset  # noqa: E402
from adasam.utils import set_seed  # noqa: E402
from unet import UNet  # noqa: E402


class AugmentedDataset(Dataset):
    def __init__(self, dataset: Dataset, seed: int) -> None:
        self.dataset = dataset
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index]
        image, mask = sample["image"], sample["masks"]
        if self.rng.random() < 0.5:
            image, mask = image.flip(-1), mask.flip(-1)
        if self.rng.random() < 0.5:
            image, mask = image.flip(-2), mask.flip(-2)
        return {**sample, "image": image, "masks": mask}


def dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    probabilities = logits.softmax(dim=1)
    target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    probabilities, target_one_hot = probabilities[:, 1:], target_one_hot[:, 1:]
    intersection = (probabilities * target_one_hot).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def stratified_split(dataset: NEUSegDataset, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    groups = {class_id: [] for class_id in range(1, 4)}
    for index in range(len(dataset)):
        mask = dataset[index]["masks"].squeeze(0)
        counts = torch.bincount(mask.flatten(), minlength=4)[1:]
        groups[int(counts.argmax()) + 1].append(index)
    rng = random.Random(seed)
    train_indices, val_indices = [], []
    for indices in groups.values():
        rng.shuffle(indices)
        split = max(1, round(len(indices) * val_fraction))
        val_indices.extend(indices[:split])
        train_indices.extend(indices[split:])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    inter = torch.zeros(4, dtype=torch.float64)
    union = torch.zeros(4, dtype=torch.float64)
    for batch in tqdm(loader, desc="evaluate", leave=False):
        prediction = model(batch["image"].to(device)).argmax(dim=1).cpu()
        target = batch["masks"].squeeze(1)
        for class_id in range(4):
            pred_class, target_class = prediction == class_id, target == class_id
            inter[class_id] += (pred_class & target_class).sum().item()
            union[class_id] += (pred_class | target_class).sum().item()
    ious = [float(inter[c] / union[c]) if union[c] else None for c in range(4)]
    foreground_ious = [iou for iou in ious[1:] if iou is not None]
    return {
        "mIoU_fg": sum(foreground_ious) / max(len(foreground_ious), 1),
        "per_class_iou": dict(zip(NEUSegDataset.CLASS_NAMES, ious)),
        "n_samples": len(loader.dataset),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-supervision Pytorch-UNet baseline on NEU_Seg")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "neu_seg_unet.yaml"))
    parser.add_argument("--data-root")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    for key, value in (("data_root", args.data_root), ("epochs", args.epochs), ("batch_size", args.batch_size), ("device", args.device), ("seed", args.seed), ("output_dir", args.output_dir), ("val_fraction", args.val_fraction)):
        if value is not None:
            cfg["train"][key] = value
    train_cfg = cfg["train"]
    set_seed(int(train_cfg["seed"]))
    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")
    data_root = Path(train_cfg["data_root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    full_train_ds = NEUSegDataset(data_root, split="train")
    train_indices, val_indices = stratified_split(full_train_ds, float(train_cfg["val_fraction"]), int(train_cfg["seed"]))
    train_ds = AugmentedDataset(Subset(full_train_ds, train_indices), int(train_cfg["seed"]))
    val_ds = Subset(full_train_ds, val_indices)
    test_ds = NEUSegDataset(data_root, split="test")
    train_loader = DataLoader(train_ds, batch_size=int(train_cfg["batch_size"]), shuffle=True, num_workers=int(train_cfg.get("num_workers", 0)), pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=int(train_cfg["batch_size"]), shuffle=False, num_workers=int(train_cfg.get("num_workers", 0)), pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=int(train_cfg["batch_size"]), shuffle=False, num_workers=int(train_cfg.get("num_workers", 0)), pin_memory=device.type == "cuda")
    model = UNet(n_channels=3, n_classes=4, bilinear=bool(cfg["model"].get("bilinear", True))).to(device)
    optimizer = AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(train_cfg["epochs"]))
    output_dir = Path(train_cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = {
        "seed": int(train_cfg["seed"]),
        "val_fraction": float(train_cfg["val_fraction"]),
        "train_indices": train_indices,
        "val_indices": val_indices,
        "train_sample_ids": [full_train_ds.sample_names[index] for index in train_indices],
        "val_sample_ids": [full_train_ds.sample_names[index] for index in val_indices],
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    best_score = -1.0
    epochs_without_improvement = 0
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{train_cfg['epochs']}"):
            image, target = batch["image"].to(device), batch["masks"].squeeze(1).to(device)
            logits = model(image)
            loss = F.cross_entropy(logits, target) + float(cfg["loss"].get("dice_weight", 1.0)) * dice_loss(logits, target, 4)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "validation": metrics}
        print(f"[NEU U-Net] epoch={epoch} loss={record['train_loss']:.4f} mIoU_fg={metrics['mIoU_fg']:.4f}")
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        checkpoint = {"stage": "neuseg_full_supervised_unet", "epoch": epoch, "model": model.state_dict(), "config": cfg, "metrics": record}
        torch.save(checkpoint, output_dir / "last_model.pt")
        if metrics["mIoU_fg"] > best_score:
            best_score = metrics["mIoU_fg"]
            epochs_without_improvement = 0
            torch.save(checkpoint, output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(train_cfg.get("early_stopping_patience", 10**9)):
            print(f"[NEU U-Net] early stopping at epoch={epoch}")
            break
    best_checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    test_metrics = evaluate(model, test_loader, device)
    (output_dir / "test_evaluation.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print(f"[NEU U-Net] test mIoU_fg={test_metrics['mIoU_fg']:.4f}")
    print(f"[NEU U-Net] done: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
