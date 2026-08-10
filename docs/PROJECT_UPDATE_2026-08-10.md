# Project Update — 2026-08-10

## Scope of this summary

The chat transcript before this task is not stored in the repository or exposed in
the current context. This document therefore summarizes the verifiable project state
from the current code, documentation, and commits through `e3b5171`. It does not
invent decisions or experimental results that are absent from those sources.

## Consolidated progress

The project has moved from an early aerial few-shot/instance-segmentation framing to
a label-efficient semantic-segmentation main line. The primary implementation freezes
MobileSAM/TinyViT, exposes multi-scale P3/P4/embedding features, and trains compact
adapter, fusion, prompt, prototype, and decoder components.

Implemented since the Revision 3 method repositioning:

- NEU-Seg semantic dataset and percentage-budget training.
- LoveDA dataset, benchmark runner, adapter ablation, and prediction visualization.
- iSAID semantic dataset, checking/preparation/tiling tools, visualization, and
  iteration-based training control.
- synchronized basic and defect-aware augmentation.
- DAPG v1, spatial DAPG v2, and frequency-aware v3 prompt generators.
- Defect Prototype Memory with compactness supervision.
- lightweight and boundary-aware decoders.
- multi-scale feature selection, pre/post-fusion adapters, hierarchical alternatives,
  SCSR variants, task-utility routing, and semantic representation budgets.
- efficiency profiling, multiseed runners, routing statistics, and focused tests.

## Current default path

The reusable model is `adasam.models.LabelEfficientSAM`. Its default configuration is
frozen MobileSAM, P3+P4+embedding features, pre-fusion CAT adaptation, hierarchical
fusion, representation budget 3, and the lightweight semantic decoder. Prompt,
prototype, boundary, and sparse-routing components are opt-in experiments.

Dataset-specific training is handled by:

| Dataset | Script | Main comparison |
|---|---|---|
| NEU-Seg | `tools/train_segmentation.py` | low-label MobileSAM variants |
| LoveDA | `tools/train_loveda.py` | U-Net / MobileSAM / proposed model |
| iSAID | `tools/train_isaid.py` | U-Net / MobileSAM / proposed model |

## Decisions that remain in force

1. Compare methods at identical labeled-image budgets and fixed validation splits.
2. Keep official test data out of model selection and threshold tuning.
3. Report exact image counts alongside percentage or K-shot labels.
4. Treat full supervision as reference context, not a fair low-data competitor.
5. Record accuracy and efficiency together: mIoU/Dice plus parameters, memory,
   training time, latency, and FPS under a fixed timing protocol.
6. Label all architectural benefits as hypotheses until multiseed controlled results
   and artifacts are registered.

## Evidence status

The repository contains implementation tests and preliminary table infrastructure,
but the experiment registry does not yet identify a complete, frozen, multiseed paper
result with manifest, checkpoint, commit, environment, and timing record. Accordingly:

- no claim that a variant improves mIoU is currently established here;
- no real-time, speedup, memory, or label-efficiency headline is established here;
- ablations must isolate one factor at a time before combining modules.

## Next execution sequence

1. Freeze dataset manifests and fixed validation splits for each benchmark.
2. Run smoke tests for the three dataset entry points and archive resolved configs.
3. Establish U-Net and frozen-MobileSAM baselines at the same annotation budgets.
4. Screen fusion/routing, prompt, prototype, boundary, and augmentation components
   independently on seed 42.
5. Promote only meaningful variants to at least three seeds.
6. Register checkpoints, commit hashes, environment, and efficiency measurements in
   `EXPERIMENT_MANIFEST.md` before using results in a paper table.
