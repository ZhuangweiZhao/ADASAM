# Candidate Core Contributions — User Confirmation Required

## A. Controlled empirical contribution (recommended)

**Core statement:** Under a controlled multi-budget, multi-seed study of industrial
defect segmentation, the dominant gains in low-label regimes arise from label-
preserving augmentation and frozen foundation features rather than the current
dynamic prompt-generation module; AdaSAM turns these positive and negative findings
into an evidence-based adaptation protocol.

**Reviewer payoff:** A practitioner learns which interventions reliably improve
label efficiency and which apparent sparse/adaptive gains do not survive controlled
comparison.

**Boundary:** This is an empirical/protocol contribution, not a claim that AdaSAM is
SOTA. It still requires fixed-split reruns and complete provenance for publication.

## B. Adaptive hierarchical retention contribution

**Core statement:** Adaptive selection of frozen MobileSAM hierarchical detail
features consistently improves segmentation and small-object IoU over random
retention at a fixed 25% representation budget across three Vaihingen seeds.

**Reviewer payoff:** The result suggests that where detail is retained matters more
than nominal sparsity alone.

**Boundary:** No acceleration or broad generalization claim; current evidence is one
dataset and approximately one-point mIoU gain.

## C. Method-centric AdaSAM contribution (not yet evidence-complete)

**Core statement:** A prompt/prototype/decoder-side sparse adaptation system improves
the accuracy-annotation-compute Pareto frontier of frozen MobileSAM.

**Why it cannot yet be confirmed:** DAPG gains are small or negative at some budgets,
efficiency timing is not comparable, sparse retention has no measured FPS benefit,
and key LoveDA studies are single-seed.

## Requested decision

Choose **A**, **B**, or **C with additional experiments before drafting**. You may
also revise the wording. No `confirmed_contribution.md` or manuscript body will be
created until this decision is explicit.
