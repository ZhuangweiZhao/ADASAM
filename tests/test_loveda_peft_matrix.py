from __future__ import annotations

from argparse import Namespace

from tools.run_loveda_peft_matrix import command_for


def args() -> Namespace:
    return Namespace(
        data_root="/data/LoveDA",
        checkpoint="/weights/mobile_sam.pt",
        output_dir="/runs/peft",
        batch_size=8,
        num_workers=4,
        epochs=100,
        device="cuda",
    )


def test_peft_matrix_lora_command_is_pure_lora() -> None:
    command = command_for(args(), "lora", ratio=10, seed=42)
    assert command[command.index("--model") + 1] == "mobilesam"
    assert command[command.index("--adapter") + 1] == "none"
    assert command[command.index("--lora-rank") + 1] == "4"
    assert command[command.index("--label-ratio") + 1] == "10"


def test_peft_matrix_full_ft_uses_lower_backbone_lr() -> None:
    command = command_for(args(), "full_ft", ratio=5, seed=123)
    assert command[command.index("--model") + 1] == "mobilesam_finetune"
    assert command[command.index("--backbone-lr-multiplier") + 1] == "0.1"
    assert command[command.index("--seed") + 1] == "123"
