# AdaSAM

**MobileSAM-based Few-shot Aerial Instance Segmentation with Adaptive Sparse Computation.**

A clean-slate research codebase (rebuilt from `AdaTile-FastSAM`) where the **only backbone is
MobileSAM**. Few-shot adaptation follows the *prototype → similarity peaks → point prompts → SAM
MaskDecoder* paradigm (PerSAM / Matcher family), so per-instance masks come straight out of SAM's
promptable decoder. Evaluated under the **frozen Evaluation Protocol V3** (COCO AP, one-to-one
instance matching, no union masks).

## Design principles

Single Responsibility · Open-Closed · KISS · YAGNI. Every module has a fixed input/output contract
and maps to exactly one paper section. **No legacy compatibility branches** (`if decoder=="..."`).

```
adasam/
├── backbone/    MobileSAMBackbone: image → {"image_embedding":[B,256,64,64]}   (frozen)
├── prototype/   PrototypeBuilder / PrototypeMemory / Matcher (sim → point prompts)
├── decoder/     PromptMaskDecoder: (embedding, prototype) → (masks, scores)
├── datasets/    ISAIDInstanceDataset + EpisodeSampler (scene-disjoint K-shot)
├── losses/      focal(eps=1e-4,γ=5.0) + dice + combined
├── metrics/     V3 FROZEN core: instance_match.py + coco_eval.py (ported verbatim)
├── trainer/     single Trainer.train()
├── evaluator/   evaluate.py → instance_metrics.json (V3 schema)
├── logging/     crash-safe structured logging
├── config/      ExperimentConfig + Recorder
└── utils/       seed, transforms (SAM 1024² preprocessing)
```

## Backbone

MobileSAM is vendored under `thirdparty/MobileSAM/` (source) with `weights/mobile_sam.pt` (~40 MB,
gitignored). Built via `mobile_sam.sam_model_registry["vit_t"]`.

## Data & evaluation

- Data (iSAID Instance Few-Shot Split, 896² COCO tiles) is **referenced by path** via
  `configs/base.yaml:data_root`, not copied.
- All paper numbers must come from `tools/evaluate.py` (the single V3 evaluator). Protocol integrity
  is guarded by `tests/test_instance_match.py` + `tests/test_protocol_audit.py`.

## Quick start

```bash
pip install -e ".[dev]"
pytest tests/ -q                     # protocol + metric unit tests
python tools/train.py    --fold 0 --k-shot 5 --epochs 50
python tools/evaluate.py --checkpoint <ckpt> --seed 42   # → instance_metrics.json
```
