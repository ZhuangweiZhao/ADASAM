# Confirmed Contribution

## Core Contribution

| Field | Content |
|---|---|
| Main contribution statement | Under a controlled multi-budget, multi-seed study of industrial defect segmentation, the dominant gains in low-label regimes arise from label-preserving augmentation and frozen foundation features rather than the current dynamic prompt-generation module; AdaSAM turns these positive and negative findings into an evidence-based adaptation protocol. |
| Contribution type | new analysis-or-benchmark |
| One-sentence reviewer payoff | The paper gives practitioners a reproducible basis for deciding whether to spend scarce annotation and compute budgets on augmentation, frozen-feature adaptation, or additional architectural complexity. |

## Why This Contribution Is Needed

| Field | Content |
|---|---|
| Field problem | Foundation segmentation models make adaptation attractive when dense pixel labels are expensive, but low-label results are sensitive to data, augmentation, model capacity and evaluation protocol. |
| Specific gap | Existing comparisons do not cleanly attribute low-label gains across annotation budget, label-preserving augmentation and lightweight adaptation under one repeated protocol. |
| Concrete challenge | Architectural changes and training recipes interact with small labeled subsets, while inconsistent timing boundaries and single-seed studies make modest gains easy to misattribute. |
| Why prior work leaves it unresolved | Standard SAM adaptation studies emphasize a proposed module; benchmark tables often compare different splits and recipes; sparse feature studies may report nominal retention without measuring end-to-end speed. |

## How This Paper Responds

| Field | Content |
|---|---|
| Design response | Construct a matched budget-by-budget matrix spanning U-Net, frozen MobileSAM adaptation and DAPG-v2, with and without synchronized augmentation, then use paired seeds and cross-dataset survival tests to separate stable effects from screening observations. |
| Evidence required | A fixed split and nested label budgets; at least three seeds; machine-readable metrics; archived checkpoints; resolved configs and commit; matched timing, memory and parameter measurements; component and augmentation ablations. |
| Evidence available | The current NEU-Seg matrix contains 108 runs across three models, six budgets, two augmentation modes and three seeds; analysis tables report mean/std foreground mIoU and paired gains. Vaihingen provides a three-seed adaptive-retention screening study; LoveDA supplies budget, layer and retention screening groups. |
| Evidence missing | The current bundles are screening rather than paper-ready: archived checkpoints, complete hardware/software provenance, corrected efficiency measurements and fixed-protocol reruns are incomplete; LoveDA budget groups are mostly single-seed. |

## Claim Boundary

| Field | Content |
|---|---|
| Strong claims allowed | In the evaluated NEU-Seg matrix, basic synchronized augmentation produces larger low-label gains than the observed DAPG-v2 paired gain; DAPG-v2 does not show a uniformly positive advantage across budgets; the current evidence supports a protocol and prioritization insight, not universal superiority. |
| Claims to soften or avoid | Do not claim SOTA, real-time performance, 10x speedup, full-supervision equivalence, causal superiority of frozen features, or cross-dataset generalization until fixed-protocol evidence and profiling are complete. |
| Novelty risk | Reviewers may view this as an engineering benchmark or a repackaging of standard ablations. The answer must be a transparent, repeated, claim-driven protocol with negative results and released evidence artifacts, not a new-module novelty claim. |
| Significance risk | The findings may be specific to industrial defect segmentation and MobileSAM. State the scope explicitly and frame broader implications as hypotheses for future controlled studies. |
