# Magnitude-Teacher Survival Experiment

## Single hypothesis

A scale-specific predictor that reads only the final MobileSAM embedding can learn
the Magnitude Top-25% P3/P4 masks closely enough to recover Magnitude's segmentation
benefit, while avoiding dense P3/P4 lateral projections at inference.

This is one experiment, not a new ablation matrix:

- Dataset: LoveDA.
- Label ratio: 10% (202 labeled training images under the fixed split).
- Seed: 42; validation seed: 42.
- Spatial retention: exactly 25% for P3 and P4.
- Representation budget: embedding plus both detail levels.
- Resolution: 1024 image and SAM input.
- Epochs: 100; basic augmentation; all other optimizer settings unchanged.

## Predictor and teacher

The student is a depthwise 3x3 convolution, GroupNorm, GELU, and a pointwise 1x1
convolution with two output maps, one each for P3 and P4. It adds roughly one thousand
parameters. Its only input is the projected final embedding.

During training, dense P3/P4 projections produce Magnitude importance maps. Their
exact Top-25% masks supervise the student with class-balanced BCE plus soft Dice.
The teacher is detached and receives no gradient. Segmentation loss and distillation
loss are optimized jointly with distillation weight 1.0.

During validation and test, teacher maps are not computed. Student Top-25% masks are
predicted first, then P3/P4 lateral projections execute only at retained positions.

## Cloud command

Run from the repository root:

```bash
python tools/run_loveda_magnitude_teacher_survival.py \
  --data-root /root/autodl-tmp/LoveDA \
  --checkpoint weights/mobile_sam.pt \
  --reference-root runs/loveda_budget_study/compression \
  --output-dir runs/loveda_magnitude_teacher_survival \
  --epochs 100 --batch-size 2 --num-workers 8 \
  --image-size 1024 --sam-image-size 1024
```

The runner trains exactly one model, profiles it, creates the three visualization
families, reads the existing 25% seed-42 references, and writes:

```text
runs/loveda_magnitude_teacher_survival/survival_verdict.json
```

## Pre-registered decision gates

All four gates must pass:

1. At least +1.0 mIoU point over the old Adaptive model.
2. Within 2.0 mIoU points of the dense Magnitude teacher.
3. Teacher/student positive-mask IoU at least 0.50 at the selected epoch.
4. Higher mIoU than Random at the same 25% budget.

If any gate fails, the output is `STOP_OR_REDESIGN`; do not expand to more budgets or
seeds. If all pass, the output is `SURVIVE`; only then repeat seeds 123 and 456.

Measured FPS, FLOPs, and memory remain descriptive in this survival test. They must be
reported, but are not a gate because the sparse gather/scatter implementation may be
limited by framework overhead. No end-to-end acceleration claim is allowed unless the
measured profiler results support it.
