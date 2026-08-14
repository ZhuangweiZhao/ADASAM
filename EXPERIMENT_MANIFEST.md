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
| NEU-B0 | NEU-Seg | U-Net low-data baseline | 1–100% | 42/123/456 | screening (3-seed runs complete, no checkpoint archived) |
| NEU-B1 | NEU-Seg | frozen MobileSAM + lightweight decoder | 1–100% | 42/123/456 | screening (3-seed runs complete, no checkpoint archived) |
| NEU-A0 | NEU-Seg | CAT + hierarchical fusion | 1–100% | 42/123/456 | screening (3-seed runs complete, no checkpoint archived) |
| NEU-A1 | NEU-Seg | isolated prompt/prototype/augmentation/boundary ablations | screening | 42 | pending |
| LVD-B0 | LoveDA | U-Net / frozen MobileSAM baselines | 1–100% | 42 | screening (single seed) |
| LVD-A0 | LoveDA | adapter, fusion, SCSR and semantic-budget study | screening | 42 (core: 42/123/456) | screening (single-seed ablations; budget/layer/magnitude groups 3-seed) |
| VHN-A0 | Vaihingen | LoRA + random/magnitude/adaptive 25% retention | 25% | 42/123/456 | screening (3-seed survival study, verdicts REDESIGN/GO/GO) |
| ISA-B0 | iSAID | U-Net / frozen MobileSAM baselines | 100% | 42 | screening (single run) |
| ISA-A0 | iSAID | proposed semantic model, iteration-controlled | screening | 42 | pending |

`pending` means that the code path may exist, but a complete evidence bundle has not
been registered. `screening` means runs exist with machine-readable metrics, but the
bundle is incomplete (missing archived checkpoint, missing multi-seed repetition, or
missing frozen protocol record) and must not be cited as a paper result.

## Registered evidence bundles (as of 2026-08-14)

### NEU-Seg — 108-run multiseed matrix (complete)

- Source: `runs/云服务器/logs/multiseed_summary.json` (aggregates) + per-run
  `metrics.json` + train logs under `runs/云服务器/results/neu_seg/budget_scsr/`
  and `runs/云服务器/legacy/runs_no_pt/`.
- Protocol: NEU_Seg, validation fraction 0.20 (validation seed 42), 100 epochs,
  models {unet, mobilesam, dapg} × augmentation {none, basic} × ratios
  {1, 5, 10, 20, 50, 100} × seeds {42, 123, 456}; planned 108, completed 108.
- Headline (3-seed mean, basic aug): mIoU @1% 0.682 (mobilesam) / 0.676 (dapg) /
  0.638 (unet); @100% 0.849 (dapg) / 0.848 (unet) / 0.846 (mobilesam). Basic
  augmentation is the largest single factor (+4.8 to +12.9 mIoU points at 1%).
- Status: screening. Missing for paper-ready: archived checkpoints, hardware/software
  precision record, and FLOPs/peak-memory measurement.

### LoveDA — semantic-budget and ablation groups

- Semantic budget K ∈ {1, 2, 3}: ratios 5/10/20%, seed 42. mIoU strictly monotonic
  K=3 > K=2 > K=1 at every ratio (e.g. 5%: 0.395 / 0.372 / 0.331). Screening (single seed).
- Layer study (6 variants, ratio 10, seeds 42/123/456): all-levels ≈ p4_embedding
  (3-seed mean ≈ 0.395–0.396), single-scale P3 clearly worst. Screening (3-seed).
- Magnitude retention 0.25/0.50/0.75/1.00 (ratio 10, seeds 42/123/456): mIoU means
  0.408 / 0.405 / 0.414 / 0.417; FPS flat ≈ 88–89. Null result for the sparse-speed
  claim. Screening (3-seed).
- Adapter / fusion / calibration / base-model ablations: single seed 42, winners flip
  with budget — not usable for claims until multi-seed.
- Status: screening overall.

### Vaihingen — adaptive-retention survival study (3-seed)

- Protocol: ISPRS Vaihingen 512² tiles, 6 classes (value 6 = ignore), 25% detail
  retention, LoRA rank 4, 32 epochs, batch 1, grad accum 2. Policies:
  random / magnitude / adaptive × seeds {42, 123, 456}.
- Results: verdicts seed42 REDESIGN (mIoU +0.92pt vs random, < 1pt gate), seed123 GO,
  seed456 GO. Adaptive consistently beats random on mIoU (+0.9–1.2pt) and Small IoU
  (+0.020–0.029) across all three seeds.
- Artifacts: `runs/vaihingen_adaptive_survival*/survival_summary.json` + metrics.json.
- Status: screening. Missing for paper-ready: archived checkpoints, commit pin,
  efficiency protocol record.

### iSAID

- Single run: frozen MobileSAM, 100% labels, 896², 160k iterations, seed 42.
  mIoU 0.473, mIoU_fg 0.438, Boundary_F1 0.557, FPS 148. Screening (single run).

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
