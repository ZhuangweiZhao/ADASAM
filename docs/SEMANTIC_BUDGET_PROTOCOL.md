# Semantic Budget Experiment Protocol

This document freezes the evaluation protocol for the semantic-conditioned budget
mechanism requested on 2026-08-10. Results produced under a different protocol must
be labeled separately.

## Method definition

The MobileSAM/TinyViT encoder remains frozen. `semantic_budget` always retains the
final embedding as a semantic anchor, selects zero, one, or two detail levels under
`representation_budget`, and selects an exact spatial fraction of each active detail
map under `feature_retention_ratio`.

Spatial policies:

| Policy | Importance source | Purpose |
|---|---|---|
| `adaptive` | learned anchor-conditioned score plus cross-level evidence | proposed image-adaptive selection |
| `static` | one train-set average importance template shared by all images | Average Static Mask control |
| `magnitude` | local absolute difference from the semantic anchor | non-learned magnitude control |
| `random` | random spatial scores at the same exact ratio | random-mask control |

The external static template must be generated with
`tools/export_static_importance.py` from exactly the same labeled LoveDA training
subset as its paired adaptive run. Unlabeled images, internal validation, and official
validation/test images must never contribute to it.

## Fixed data and optimization controls

- Dataset: LoveDA, official Train and Val domains.
- Internal validation: fixed 20% of Train, `validation_seed=42`.
- Label subset: nested subset of the remaining Train pool, paired by experiment seed.
- Official Val: one final test after checkpoint selection.
- Resolution: image and SAM input both 1024 unless explicitly reported otherwise.
- Augmentation: synchronized `basic`, training only.
- Epochs: 100; AdamW and all remaining optimization defaults are identical.
- Screening seed: 42. Layer and compression conclusions require 42/123/456.
- Low-label budgets: 5%, 10%, 20%, 100%, with exact image counts recorded.

## Five experiment groups

1. `layer`: P3, P4, embedding, and controlled layer combinations. Report mIoU,
   Boundary F1, and Small/Medium/Large region IoU.
2. `calibration`: Concat, Sum, dataset-static learned weights, image-conditioned
   weights, and local dynamic calibration.
3. `compression`: Random, Magnitude, Average Static, and Adaptive policies at exactly
   25/50/75/100% retained detail positions.
4. `budget`: Adaptive policy at 25/50/75/100%, with dense FLOPs, measured FPS, peak
   memory, and active-detail representation fraction.
5. `low_label`: U-Net, frozen dense MobileSAM, and adaptive budget allocation at
   5/10/20/100%, three seeds.

Run the complete dependency-aware matrix on the cloud server:

```bash
python tools/run_loveda_budget_study.py \
  --study all \
  --data-root /path/to/LoveDA \
  --checkpoint /path/to/mobile_sam.pt \
  --output-dir runs/loveda_budget_study \
  --epochs 100 --batch-size 2 --num-workers 8 \
  --image-size 1024 --sam-image-size 1024 \
  --screening-seeds 42 \
  --final-seeds 42 123 456
```

The runner is restartable and skips only runs containing the requested number of
epochs. Use `--rerun-completed` only when intentionally replacing results.

## Required visual evidence

After selecting a representative adaptive checkpoint:

```bash
python tools/visualize_budget_allocation.py \
  --data-root /path/to/LoveDA \
  --checkpoint /path/to/mobile_sam.pt \
  --model-checkpoint runs/.../best_model.pt \
  --metrics runs/.../metrics.json \
  --output-dir runs/.../visualizations \
  --feature-retention-ratio 0.5
```

The command produces:

- same-image P3/P4/embedding response panels;
- Original/GT/Importance/Retained/Prediction allocation panels;
- Small/Medium/Large and background/interior/boundary retention plots;
- a JSON visualization manifest with actual retention ratios and selected levels.

## Metric definitions

Small, medium, and large refer to 8-connected foreground GT components occupying at
most 0.1%, at most 1%, and more than 1% of the resized image. Region IoU is computed
against same-class prediction inside the component bounding box padded by two pixels.
It is a conditioned diagnostic and must not be relabeled as COCO instance IoU/AP.

`Boundary_F1` uses semantic label transitions with a two-pixel tolerance at 1024.
Retention statistics are computed at the routing resolution after nearest-neighbor
projection of GT labels.

## Claim gate

`active_detail_representation_fraction` is a representation-retention proxy, not an
executed-FLOPs measurement. Adaptive/static/random policies execute the P3/P4 1x1
lateral projections only at retained positions; the frozen backbone, anchor path,
refinement, and classifier remain dense. The profiler therefore reports THOP's
executed module graph plus the explicitly counted sparse lateral projections, as well
as measured latency/FPS and memory. A paper may claim better representation allocation
when accuracy/retention evidence supports it. It may claim end-to-end acceleration
only if measured FLOPs, latency/FPS, and peak memory improve. Do not substitute the
retention proxy or lateral-only saving for an end-to-end speedup.
