# Section Blueprints

## Title

Use a bounded empirical title: *What Drives Label-Efficient Adaptation of
Foundation Segmentation Models? A Controlled Study of Augmentation and Lightweight
Adaptation*. Do not put AdaSAM or real-time in the title until method evidence is
stronger.

## Abstract

Problem -> attribution gap -> controlled matrix -> strongest augmentation and DAPG
observations -> evidence-based protocol payoff -> screening limitation.

## 1 Introduction

Establish annotation cost, foundation adaptation and the attribution problem. Review
existing SAM/adapter/prompt directions. State the single contribution and preview
the 108-run evidence. End with explicit claim boundaries.

## 2 Evaluation Question and Protocol

Define label-efficient semantic segmentation, datasets, nested budgets, seeds,
augmentation controls, models, metrics and timing boundaries. Separate paper-ready
requirements from current screening status.

## 3 Methods and Interventions

Describe frozen MobileSAM, U-Net reference, DAPG-v2, synchronized augmentation and
the adaptive-retention auxiliary study. Explain each intervention as a factor in the
attribution design, not as an unsupported innovation claim.

## 4 Results

4.1 Budget scaling; 4.2 augmentation attribution; 4.3 DAPG paired effects and
variance; 4.4 cross-dataset screening; 4.5 negative sparse-retention result.
Every subsection maps to one claim in `claim_register.md`.

## 5 Discussion

Explain why label-preserving variation may dominate small-module changes under scarce
labels, distinguish observation from mechanism, discuss protocol implications and
state novelty/significance risks.

## 6 Limitations and Next Experiments

List missing checkpoints, provenance, fixed timing, LoveDA seeds and external citation
verification. Give a minimal paper-ready rerun plan.

## 7 Conclusion

Restate the bounded empirical payoff and avoid universal or deployment claims.
