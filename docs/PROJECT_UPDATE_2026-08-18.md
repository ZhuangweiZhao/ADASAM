# Project Update — 2026-08-18

## Current Paper Direction

The project has converged on **label-efficient and parameter-efficient adaptation of
MobileSAM for high-resolution remote-sensing semantic segmentation**. The active line is:

```text
Frozen MobileSAM + LoRA rank 4 (qkv + proj) + CRHC
```

The method is **Class-Conditioned Regional Hierarchical Calibration (CRHC)**. The
project no longer treats budgeted sparse computation, FLOP reduction, or active sample
selection as primary contributions.

## CRHC Definition

CRHC obtains coarse regional semantic probabilities from high-level features, learns
class-by-level preferences over P3, P4, and embedding, maps those preferences to
pixel-level weights, and applies a residual correction to Dense Sum:

```text
class prior -> regional probability -> pixel-level hierarchy weights -> feature calibration
```

All three feature levels are still computed. CRHC is feature calibration/fusion, not
dynamic computation allocation, sparse inference, or FLOP reduction.

## LoveDA Evidence

Protocol: fixed 20% internal validation from the 2,522-image training split, official
Val as test, 512 x 512 input, seed 42, remote-strong augmentation, class-balanced CE,
Lovasz weight 0.5, and cosine schedule.

| Labels | LoRA + Dense Sum | LoRA + CRHC | Difference |
|---:|---:|---:|---:|
| 5% | 45.83 | 45.54 | -0.29 |
| 10% | 46.31 | 46.93 | +0.62 |
| 20% | 49.21 | 49.61 | +0.40 |

CRHC uses 347,845 trainable parameters versus 6,368,131 for full MobileSAM
fine-tuning, about 5.46% of the full-finetuning trainable parameter count. At 20%
labels, it is 0.58 points below the label-matched full-finetuning result (50.19%).
The separate 100%-label full-supervision reference is 53.98% and must not be mixed
with that comparison.

These are single-seed screening results. The 5% decrease means CRHC cannot yet be
claimed to improve every annotation budget.

## HRCS Negative Experiment

The label-free sample-selection test used 101 images (5%), the same LoRA + Dense Sum
adaptation, and seed 42:

| Selection | mIoU | Difference from Random |
|---|---:|---:|
| Random | 45.83 | 0.00 |
| Embedding k-center | 42.71 | -3.13 |
| P3/P4/Embedding HRCS | 45.95 | +0.12 |

HRCS improved Water (+3.23 IoU points) and Forest (+2.71), but reduced Road,
Building, and Agriculture. It improved worst-case representation coverage while
worsening mean nearest-neighbor distance. This shows that feature-space diversity
alone does not necessarily equal task-effective supervision in the tested protocol.
HRCS is exploratory/negative evidence, not a second core contribution.

## Cross-Dataset Validation

Vaihingen is the next validation dataset. The minimum fair test is 10% labels, seed 42,
LoRA rank 4, and a direct comparison between Dense Sum and `regional_semantic`. The
Vaihingen entry point uses its own CE/Dice protocol and does not yet reproduce
LoveDA's Lovasz and coarse-auxiliary settings. Vaihingen results must therefore be
described as cross-dataset mechanism validation under the Vaihingen protocol.

## Claims Allowed Now

- CRHC is a class-conditioned regional hierarchical feature-calibration mechanism.
- In LoveDA single-seed screening, CRHC improves LoRA at 10% and 20% labels, with
  gains partly concentrated in Water and Forest.
- LoRA + CRHC provides a strong parameter-efficiency comparison against label-matched
  full fine-tuning.
- Frozen-feature diversity alone was not a reliable proxy for segmentation supervision
  value in the tested HRCS setting.

## Claims Still Prohibited

- CRHC reduces FLOPs or performs sparse/dynamic computation.
- CRHC is stable across seeds or improves all label budgets.
- Learned class-level preferences are universal causal scale laws.
- HRCS improves label efficiency.
- 49.61% is only 0.58 points below the 100%-label 53.98% result.
- The method is SOTA on LoveDA without verified protocol-matched literature results.

## Next Experiments

1. Vaihingen 10% seed 42: LoRA + Sum versus LoRA + CRHC.
2. If positive, add Vaihingen seeds 123 and 456.
3. On LoveDA, run 10% seeds 123 and 456 for both LoRA + Sum and LoRA + CRHC.
4. Add focused CRHC ablations: global shared weighting, no prototype, and uniform or
   shuffled class-level hierarchy weights.
5. Register paper-facing runs with checkpoint, commit, environment, and timing
   artifacts before drafting quantitative claims.

## Verification

The current implementation passes 205 tests with 6 skips. HRCS selection and manifest
validation are implemented, but HRCS is not scheduled for expansion unless explicitly
reopened.
