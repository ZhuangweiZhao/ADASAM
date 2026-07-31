"""Low-data U-Net baseline using the same K-shot manifest as Stage2."""

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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from train_neu_seg import (  # noqa: E402
    AugmentedDataset,
    UNet,
    dice_loss,
    evaluate,
)
from adasam.datasets import NEUSegDataset  # noqa: E402
from adasam.utils import set_seed  # noqa: E402


def stratified_support_split(
    dataset: NEUSegDataset, indices: list[int], val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split the shared manifest while preserving dominant foreground-class coverage."""
    groups = {class_id: [] for class_id in (1, 2, 3)}
    for index in indices:
        mask = dataset[index]["masks"].squeeze(0)
        dominant = int(torch.bincount(mask.flatten(), minlength=4)[1:].argmax()) + 1
        groups[dominant].append(index)
    rng = random.Random(seed)
    train_indices, val_indices = [], []
    for group in groups.values():
        rng.shuffle(group)
        val_count = 0 if len(group) <= 1 else max(1, round(len(group) * val_fraction))
        val_indices.extend(group[:val_count])
        train_indices.extend(group[val_count:])
    if not val_indices and train_indices:
        val_indices.append(train_indices.pop())
    return train_indices, val_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-data U-Net baseline on NEU_Seg")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "neu_seg_unet.yaml"))
    parser.add_argument("--stage1-ckpt", required=True, help="Checkpoint containing kshot_manifest")
    parser.add_argument("--data-root")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--manifest", help="K-shot manifest JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    overrides = {
        "data_root": args.data_root, "epochs": args.epochs, "batch_size": args.batch_size,
        "device": args.device, "seed": args.seed, "output_dir": args.output_dir,
        "val_fraction": args.val_fraction,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg["train"][key] = value
    train_cfg = cfg["train"]
    set_seed(int(train_cfg["seed"]))
    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")
    data_root = Path(train_cfg["data_root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    full_train = NEUSegDataset(data_root, split="train")
    test_ds = NEUSegDataset(data_root, split="test")
    checkpoint = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
    manifest = checkpoint.get("kshot_manifest")
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if not manifest:
        raise ValueError("Stage1 checkpoint does not contain kshot_manifest")
    by_name = {name: index for index, name in enumerate(full_train.sample_names)}
    support_indices = [by_name[name] for name in manifest["sample_ids"] if name in by_name]
    if len(support_indices) != len(manifest["sample_ids"]):
        raise ValueError("K-shot manifest contains samples absent from NEU_Seg training split")
    train_indices, val_indices = stratified_support_split(
        full_train, support_indices, float(train_cfg.get("val_fraction", 0.2)), int(train_cfg["seed"])
    )
    train_ds = AugmentedDataset(Subset(full_train, train_indices), int(train_cfg["seed"]))
    val_ds = Subset(full_train, val_indices)
    loader_kwargs = {"batch_size": int(train_cfg["batch_size"]), "num_workers": int(train_cfg.get("num_workers", 0)), "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    model = UNet(3, 4, bilinear=bool(cfg["model"].get("bilinear", True))).to(device)
    optimizer = AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(train_cfg["epochs"]))
    output_dir = Path(train_cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    split = {"k_shot": manifest["k_shot"], "seed": train_cfg["seed"], "support_ids": manifest["sample_ids"], "train_ids": [full_train.sample_names[i] for i in train_indices], "val_ids": [full_train.sample_names[i] for i in val_indices]}
    (output_dir / "split_manifest.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    best_score, stale = -1.0, 0
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
        val_metrics = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "validation": val_metrics}
        print(f"[NEU Low-data U-Net] epoch={epoch} loss={record['train_loss']:.4f} val_mIoU_fg={val_metrics['mIoU_fg']:.4f}")
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        checkpoint_data = {"stage": "neuseg_low_data_unet", "epoch": epoch, "model": model.state_dict(), "config": cfg, "kshot_manifest": manifest, "metrics": record}
        torch.save(checkpoint_data, output_dir / "last_model.pt")
        if val_metrics["mIoU_fg"] > best_score:
            best_score, stale = val_metrics["mIoU_fg"], 0
            torch.save(checkpoint_data, output_dir / "best_model.pt")
        else:
            stale += 1
        if stale >= int(train_cfg.get("early_stopping_patience", 20)):
            break
    best = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics = evaluate(model, test_loader, device)
    (output_dir / "test_evaluation.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print(f"[NEU Low-data U-Net] test_mIoU_fg={test_metrics['mIoU_fg']:.4f}")


if __name__ == "__main__":
    main()
