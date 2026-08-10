# Experiment Manifest — AdaSAM

This is the traceability registry for current label-efficient semantic-segmentation
experiments. A paper number is admissible only when its row identifies the data/split
manifest, seed, resolved config, checkpoint, commit, environment, and result artifact.

## Required evidence

| Field | Requirement |
|---|---|
| Task | Dataset, classes, label budget, exact labeled-image count |
| Split | Frozen train/validation manifest; official test excluded from selection |
| Repetition | Seed or independently sampled manifest; at least three for final claims |
| Model | Resolved CLI/config and trainable/total parameter counts |
| Artifact | Checkpoint plus machine-readable metrics and timing output |
| Provenance | Git commit and hardware/software/precision record |
| Metrics | foreground macro mIoU, per-class IoU, Dice/pixel accuracy as applicable |
| Efficiency | wall time, peak memory, end-to-end latency/FPS with timing method |

## Registry

| ID | Dataset | Configuration | Budget | Seeds | Evidence status |
|---|---|---|---|---|---|
| NEU-B0 | NEU-Seg | U-Net low-data baseline | 1–100% | 42/123/456 | pending |
| NEU-B1 | NEU-Seg | frozen MobileSAM + lightweight decoder | 1–100% | 42/123/456 | pending |
| NEU-A0 | NEU-Seg | CAT + hierarchical fusion | 1–100% | 42/123/456 | pending |
| NEU-A1 | NEU-Seg | isolated prompt/prototype/augmentation/boundary ablations | screening | 42 | pending |
| LVD-B0 | LoveDA | U-Net / frozen MobileSAM baselines | 1–100% | 42/123/456 | pending |
| LVD-A0 | LoveDA | adapter, fusion, SCSR and semantic-budget study | screening | 42 | pending |
| ISA-B0 | iSAID | U-Net / frozen MobileSAM baselines | 1–100% | 42/123/456 | pending |
| ISA-A0 | iSAID | proposed semantic model, iteration-controlled | screening | 42 | pending |

`pending` means that the code path may exist, but a complete evidence bundle has not
been registered. Do not infer completion from files under `runs/` or `outputs/`.

## Result record template

```text
ID:
Dataset and exact labeled-image count:
Train/validation/test manifest:
Seed:
Resolved command/config:
Checkpoint:
Metrics artifact:
Timing artifact and method:
Hardware/software/precision:
Git commit:
Status: screening | replicated | paper-ready | rejected
```

Legacy V3 instance-evaluation records belong to the historical few-shot instance
segmentation line and must not be mixed with this semantic-segmentation registry.
