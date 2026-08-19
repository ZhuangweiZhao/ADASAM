# Motivation Options After Research

## Option A — Evidence-first label-efficiency study (recommended)

Practitioners adapting foundation segmentation models under scarce pixel labels do
not know whether to invest in a new adaptation module or in stronger control of data,
augmentation and protocol. AdaSAM provides a controlled budget-by-budget analysis
showing that augmentation and frozen representations explain more of the observed
gain than the current prompt-generation variant. The paper's value is a reproducible
decision map, including negative results.

**Evidence fit:** strongest. **Required work:** rerun the decisive fixed protocol,
archive provenance and replace the invalid timing table.

## Option B — Adaptive detail-retention method

Frozen final embeddings lose spatial detail important for small structures; adaptive
retention of hierarchical features can preserve useful detail under a constrained
representation budget. The three-seed Vaihingen result is the starting evidence.

**Evidence fit:** medium. **Required work:** fixed efficiency profiling, dense and
random/magnitude baselines, another dataset and component ablations.

## Option C — Semantic-budget allocation

Label-efficient adaptation should allocate representation capacity according to the
annotation budget rather than use a fixed multi-scale recipe. The monotonic LoveDA
K-budget result motivates this direction.

**Evidence fit:** preliminary. **Required work:** three seeds at all budgets, matched
parameter/compute controls and a clear mechanism beyond adding more feature levels.

## Rejected controlling motivation

`AdaSAM is already a real-time method that matches full supervision with 5% labels`
is rejected: the current timing protocols are not comparable, sparse retention did
not improve measured FPS, and the accuracy claim is not supported across datasets.
