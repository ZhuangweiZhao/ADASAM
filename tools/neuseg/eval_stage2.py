"""Evaluate a trained NEU_Seg Stage 2 checkpoint on the official test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = Path(__file__).resolve().parent
for path in (str(_REPO_ROOT), str(_SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from train_stage2 import Stage2Trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NEU_Seg Stage 2")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage1-ckpt", help="Override embedded Stage 1 checkpoint path")
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    checkpoint_path = Path(cli.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = _REPO_ROOT / checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "neuseg_prototype_semantic_query_stage2":
        raise ValueError(f"not a prototype semantic-query checkpoint: {checkpoint_path}")
    cfg = checkpoint["config"]
    if checkpoint.get("class_thresholds"):
        cfg.setdefault("eval", {})["class_thresholds"] = checkpoint["class_thresholds"]
    if cli.data_root is not None:
        cfg.setdefault("data", {})["data_root"] = cli.data_root
    if cli.device is not None:
        cfg.setdefault("train", {})["device"] = cli.device
    if cli.max_samples > 0:
        cfg.setdefault("train", {})["val_samples"] = cli.max_samples
    else:
        cfg.setdefault("train", {})["val_samples"] = 10**9
    if cli.threshold is not None:
        cfg.setdefault("eval", {})["foreground_threshold"] = cli.threshold

    stage1_path = cli.stage1_ckpt or checkpoint.get("stage1_checkpoint")
    if not stage1_path:
        raise ValueError("Stage 1 checkpoint path is missing; pass --stage1-ckpt")
    trainer_args = argparse.Namespace(
        stage1_ckpt=stage1_path,
        steps=None,
        epochs=None,
        episodes=None,
        support_shot=None,
        seed=None,
        device=cli.device,
        data_root=cli.data_root,
        output_dir=None,
        val_samples=cli.max_samples or None,
    )
    trainer = Stage2Trainer(cfg, trainer_args)
    missing, unexpected = trainer.model.load_state_dict(checkpoint["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    metrics = trainer.validate(use_all_support=True)
    output = Path(cli.output) if cli.output else checkpoint_path.parent / "evaluation.json"
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"[NEU Stage2 Eval] saved: {output}")


if __name__ == "__main__":
    main()
