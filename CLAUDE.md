# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable + dev deps)
pip install -e ".[dev]"

# Run all tests
pytest tests/ -q

# Run a single test file or test
pytest tests/test_instance_match.py -q
pytest tests/test_instance_match.py::test_greedy_perfect_match -q

# Lint / format
ruff check adasam/ tools/ tests/
black adasam/ tools/ tests/

# Train (config from configs/base.yaml, CLI overrides)
python tools/train.py --fold 0 --k-shot 5 --epochs 50
python tools/train.py --epochs 1 --episodes 2          # smoke test

# Evaluate (all paper numbers must come from this script)
python tools/evaluate.py --checkpoint runs/.../best_model.pt --k-shot 5 --seed 42
python tools/evaluate.py --checkpoint <ckpt> --limit 5 --no-zero-shot  # smoke
```

## Architecture

AdaSAM is a **clean-slate** few-shot aerial instance segmentation framework. The single backbone is **MobileSAM** (TinyViT image encoder, frozen). Few-shot adaptation follows the PerSAM/Matcher paradigm: prototype → similarity peaks → point prompts → SAM MaskDecoder. Every paper number is produced by `tools/evaluate.py` under the frozen **Protocol V3**.

### Data flow

```
Image (896² tile, RGB)
  → preprocess_image() [adasam/utils/transforms.py] → [3, 1024, 1024] normalized
  → MobileSAMBackbone.forward() → {"image_embedding": [B, 256, 64, 64]}
  → PrototypeBuilder.build(K support embeddings + masks) → prototype [256]
  → similarity_map(embedding, prototype) → [64, 64] cosine similarity
  → Matcher.select(sim_map) → PromptPoints (top-K peaks via greedy NMS)
  → PromptMaskDecoder.decode(image_embedding, prototype, points) → low-res logits
  → upscale_logits() → per-instance masks at tile resolution (896²)
```

### Module contracts (no legacy `if` branches)

| Module | Input → Output | Trainable? |
|---|---|---|
| `adasam/backbone/` | `[B,3,1024,1024]` → `{"image_embedding":[B,256,64,64]}` | **No** — always frozen, `train()` overridden to no-op |
| `adasam/prototype/` | K support (embedding + FG mask) → L2-normalized prototype [256] | **No** — purely algorithmic (masked average pooling) |
| `adasam/decoder/` | `(image_embedding, prototype)` → `InstanceMasks(masks, scores)` | **Yes** — PrototypeAdapter (zero-init final layer) + MaskDecoder; PromptEncoder frozen |
| `adasam/losses/` | `(logits, targets)` → scalar (focal+dice+IoU-head MSE) | N/A |
| `adasam/metrics/` | `(pred_masks, gt_masks)` → TP/FP/FN, Instance mIoU, COCO AP | N/A — pure numpy/pycocotools |

Training uses **teacher forcing**: GT interior points (distance-transform peaks) as prompts → decode → supervise with focal+dice. Inference uses prototype-similarity peaks. Both share `PromptMaskDecoder.decode()` — no if-mode branches.

### Vendored third-party code

MobileSAM is vendored under `thirdparty/MobileSAM/` and injected via `sys.path` at runtime by `adasam/backbone/mobile_sam.py:_ensure_mobile_sam_on_path()`. It is NOT declared as a pip package. The weights file (`weights/mobile_sam.pt`, ~40 MB) is gitignored. Tests that depend on weights auto-skip when the file is absent.

### Evaluation Protocol V3 (frozen)

All paper numbers must come from `tools/evaluate.py`. The protocol:
1. **Instance-level, never union masks** — each GT/prediction is an independent entity
2. **One-to-one greedy matching** (by descending score) — a prediction cannot match multiple GT
3. **COCO AP** from official pycocotools via `COCOInstanceEvaluator` (the only file allowed to call `COCOeval` directly)
4. **Instance mIoU** — per-GT max-IoU prediction, averaged (intentionally does NOT enforce one-to-one; separate from greedy_match)
5. **Zero-shot** — MobileSAM everything-mode, class-agnostic, non-oracle
6. **Frozen manifest** — `evaluation_manifest_val.json` locks the query tile set across runs

The audit guard at `tests/test_protocol_audit.py` scans the entire repo for forbidden legacy patterns (oracle zero-shot, self-rolled AP, raw `COCOeval` outside the sanctioned file, non-deterministic `hash()` in sampling).

### Configuration

Single config source: `configs/base.yaml`. Read by both `tools/train.py` and `tools/evaluate.py`. CLI arguments override YAML fields. The checkpoint's embedded config is the default source during evaluation (weights path, embed_dim, matcher params are read from the checkpoint, not the yaml).

### Dataset

iSAID Instance Few-Shot Split — 896² COCO tiles, 15 classes, 3-fold base/novel splits. Data is **referenced by path** (`configs/base.yaml:data_root`), not copied. `EpisodeSampler` enforces scene-disjoint sampling (support and query from different source images) with a `min_tiles` filter. The sampler depends only on a query interface (`visible_classes`, `class_to_tiles`, `source_image_id`), decoupled from the concrete dataset.

### Logging

Structured logging system (`adasam/logging/`) — all observable values go through `get_logger(name)` with named backends (ConsoleBackend, FileBackend). No bare `print()`. File output is JSONL to `runs/` (gitignored). Backend write failures are silently caught to never crash training.

### Experiment tracking

`EXPERIMENT_MANIFEST.md` is the traceability registry. Every paper number binds: ID → Protocol → Manifest → Seed → Checkpoint → Commit. A number may enter a paper table only with Protocol=V3, a frozen Manifest, a Checkpoint, and a Commit hash.
