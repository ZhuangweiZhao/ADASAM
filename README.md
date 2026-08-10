# AdaSAM

AdaSAM is a MobileSAM-based research codebase for label-efficient semantic
segmentation. The current main line freezes the MobileSAM image encoder and studies
compact adaptation on NEU-Seg, LoveDA, and iSAID.

> Current status (2026-08-10): the implementation and experiment infrastructure are
> under active development. Model variants are implemented and covered by tests, but
> no unverified accuracy or efficiency claim should be treated as a paper result.

## Current research question

Can a frozen MobileSAM encoder, lightweight multi-scale adaptation, and a compact
semantic decoder improve the accuracy/annotation/compute trade-off over low-data
supervised baselines?

This is label-efficient semantic segmentation, not classical base-to-novel few-shot
segmentation. Full-supervision results are reference upper bounds and must not be
presented as label-budget-matched comparisons.

## Implemented system

```text
image
  -> frozen MobileSAM/TinyViT (P3, P4, embedding)
  -> optional CAT adapter
  -> multi-scale fusion / routing
  -> lightweight or boundary-aware semantic decoder
  -> semantic logits at the original image size
```

Optional experimental components include:

- DAPG prompt generators (`v1`, spatial `v2`, frequency-aware `v3`)
- Defect Prototype Memory (DPM)
- basic and defect-aware synchronized augmentation
- hierarchical, global, image-conditioned, SCSR, task-routed, and semantic-budget fusion
- boundary auxiliary supervision and gated boundary fusion
- pre-fusion or post-fusion CAT adapter placement

## Datasets and entry points

| Dataset | Primary entry point | Notes |
|---|---|---|
| NEU-Seg | `tools/train_segmentation.py` | 1/5/10/20/25/50/100% budgets |
| LoveDA | `tools/train_loveda.py` | U-Net, frozen MobileSAM, and proposed variants |
| iSAID | `tools/train_isaid.py` | epoch- or iteration-based semantic tile training |

Data and `weights/mobile_sam.pt` are referenced locally and are not committed.

## Installation and checks

```bash
pip install -e ".[dev]"
pytest tests -q
```

Example smoke runs:

```bash
python tools/train_segmentation.py --label_ratio 5 --epochs 1 --device cuda
python tools/train_loveda.py --model ours --label-ratio 5 --epochs 1 --device cuda
python tools/train_isaid.py --model ours --label-ratio 5 --data-root <tiles> \
  --output-dir runs/isaid_smoke --max-iterations 10 --eval-interval 10
```

Use `--help` on each entry point for the full variant matrix. Outputs contain the
resolved arguments, metrics, checkpoint, timing, and parameter counts where supported.

## Documentation

Start at [`docs/README.md`](docs/README.md). The current project summary is
[`docs/PROJECT_UPDATE_2026-08-10.md`](docs/PROJECT_UPDATE_2026-08-10.md), the paper
position is in [`METHOD_DESIGN.md`](METHOD_DESIGN.md), and experiment traceability is
maintained in [`EXPERIMENT_MANIFEST.md`](EXPERIMENT_MANIFEST.md).

The frozen protocol and cloud command for the semantic-budget study are in
[`docs/SEMANTIC_BUDGET_PROTOCOL.md`](docs/SEMANTIC_BUDGET_PROTOCOL.md).

Legacy few-shot/promptable-decoder modules remain in the repository for historical
experiments. They are not the default label-efficient training path described above.
