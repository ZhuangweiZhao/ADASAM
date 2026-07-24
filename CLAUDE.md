# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable + dev deps)
pip install -e ".[dev]"

# Run all tests
pytest tests/ -q

# Run a single test file or test
pytest tests/test_semantic_prior_generator.py -q
pytest tests/test_model_forward.py -q

# Lint / format
ruff check adasam/ tools/ tests/
black adasam/ tools/ tests/

# AdaSAM Two-Stage Training Protocol (iSAID-5i)
# Stage 1: Domain Adaptation (standard semantic segmentation, no few-shot)
python tools/adasam/train_stage1.py --fold 0 --epochs 50
python tools/adasam/train_stage1.py --fold 0 --epochs 1 --steps 5    # smoke

# Stage 2: Few-shot Semantic Learning (episodic, requires Stage 1 adapter)
python tools/adasam/train_stage2.py --fold 0 --k-shot 5 --epochs 50 \
    --stage1-ckpt runs/stage1_fold0_seed42/best_adapter.pt
python tools/adasam/train_stage2.py --fold 0 --k-shot 5 --epochs 1 --steps 5 \
    --stage1-ckpt runs/stage1_fold0_seed42/best_adapter.pt    # smoke

# Evaluation (iSAID-5i, FSS Benchmark protocol)
python tools/adasam/eval.py --checkpoint <ckpt> --k-shot 5               # single fold
python tools/adasam/eval.py --checkpoint <ckpt> --k-shot 5 --all-folds   # 3-fold CV
python tools/adasam/eval.py --checkpoint <ckpt> --k-shot 5 --seeds 42 123 456  # multi-seed
python tools/adasam/eval.py --checkpoint <ckpt> --k-shot 5 --max-samples 10   # smoke
python tools/adasam/eval.py --checkpoint <ckpt> --k-shot 5 --save-vis --diagnostics  # full

# NEU-SEG (multi-class segmentation, 480x640, 35 images)
python tools/neuseg/train.py --k-shot 3 --epochs 100               # train
python tools/neuseg/eval.py --checkpoint <ckpt> --k-shot 3         # evaluate
python tools/neuseg/viz.py --mode dataset                          # dataset overview
python tools/neuseg/viz.py --mode support --k-shot 3               # support/query pairs
python tools/neuseg/viz.py --mode predict --checkpoint <ckpt>      # prediction comparison
python tools/neuseg/viz.py --mode all --checkpoint <ckpt>          # all visualizations

# SAM-RSP 3-Stage Reproduction on iSAID-5i
python tools/sam_rsp/prepare.py --data-root data/iSAID-5i
python tools/sam_rsp/stage1_train.py --fold 0 --epochs 100
python tools/sam_rsp/stage2_train.py --fold 0 --shot 1 --epochs 50 \
    --stage1-ckpt runs/sam_rsp_stage1/fold0/best_model.pth
python tools/sam_rsp/download_sam.py
python tools/sam_rsp/stage3_train.py --fold 0 --shot 1 --epochs 50 \
    --stage2-ckpt runs/sam_rsp_stage2/fold0_shot1/best_model.pth \
    --sam-ckpt weights/sam_vit_h_4b8939.pth

# [DEPRECATED] Legacy instance segmentation pipeline
python tools/deprecated/train.py --fold 0 --k-shot 5 --epochs 50
python tools/deprecated/evaluate.py --checkpoint runs/.../best_model.pt --k-shot 5 --seed 42
```

## Architecture

AdaSAM is a **dual-branch semantic prior** few-shot aerial semantic segmentation framework:
- **Stage 1 (Domain Adaptation)**: MobileSAM (frozen) + CATAdapter + SegHead → standard semantic segmentation on base classes → domain-aware feature initialization.
- **Stage 2 (Few-shot Semantic Learning)**: SupportEncoder → GeometricPrior + SPG → PromptFusion → SAM Decoder. Episodic training on base classes. Novel classes inferred directly (no finetune).
- Core metrics: **mIoU** and **FB-IoU** under the **FSS Benchmark protocol**.

### Data flow (v5 — Protocol-Aligned SPG)

```
Stage 1:
  Image (256² tile, RGB)
    → MobileSAMBackbone.forward() → {"image_embedding": [B, 256, 64, 64]}
    → CATAdapter (trainable) → [B, 256, 64, 64]
    → SegHead (1×1 Conv) → [B, num_base, 256, 256] logits
    → CE + Focal + Dice (multiclass)

Stage 2:
  Support (K tiles) → MobileSAM + frozen Adapter → SupportEncoder → support_memory [M, 256]
  Query (1 tile)   → MobileSAM + frozen Adapter → query_features [1, 256, 64, 64]
                          │                              │
                          ├──→ GeometricPrior ──────────┤
                          │    (cosine similarity)       │
                          │    geometric_prior [1,C,H,W] │
                          │                              │
                          └──→ SPG ──────────────────────┤
                               (N internal probes,        │
                                aggregated to unified)    │
                               semantic_prior [1,C,H,W]   │
                               prior_mask [1,1,H,W]       │
                                                          │
                          PromptFusion ←──────────────────┘
                               │
                     dense_prompt [1,C,H,W] + sparse_token [1,C]
                               │
                          SAM Decoder (boundary refinement)
                               │
                          Fine Mask [1, 256, 256]

  Dense prompt fallback (when PromptFusion disabled):
    AdaSAMModel._build_dense_prompt() — spatial path (support features × masks)
    or legacy global path (attention-pooled support memory).

  Loss = L_main(CE+Focal+Dice on mask) + λ₁·L_prior(BCE+Dice on unified prior_mask) + λ₂·L_reg
```

### Module contracts

| Module | Input → Output | Trainable? |
|---|---|---|
| `adasam/backbone/` | `[B,3,1024,1024]` → `{"image_embedding":[B,256,64,64]}` | **No** — always frozen |
| `adasam/adapters/` | `[B,256,64,64]` → `[B,256,64,64]` | **Stage 1 only** — frozen in Stage 2 |
| `adasam/support_encoder/` | `([K,256,64,64], [K,64,64])` → support_memory `[M,256]` | **Stage 2** — TransformerEncoder + MemoryBank |
| `adasam/prompt/` (SPG) | `(query_features, support_memory, pe)` → `SPGOutput` (semantic_prior, prior_mask, prior_aux) | **Stage 2** |
| `adasam/prompt/` (GeometricPrior) | `(query_features, support_memory)` → `geometric_prior [1,C,H,W]` | **Stage 2** |
| `adasam/prompt/` (PromptFusion) | `(geometric_prior, semantic_prior)` → `(dense_prompt, sparse_token)` | **Stage 2** |
| `adasam/decoder/` | `(image_embedding, sparse_token [1,C], dense_prompt)` → `(low_res [1,1,256²], iou_pred [1,1])` | **Stage 2** — MaskDecoder; PromptEncoder frozen |
| `adasam/losses/` | `(pred, gt, prior_masks?, prior_mask?)` → `{loss, L_main, L_prior, L_reg}` | N/A |

SPG internally uses N learnable semantic probes (Mask2Former-style) but externally exposes
only unified semantic_prior + prior_mask. Dense prompt / sparse token are produced
exclusively by PromptFusion (or AdaSAMModel fallback). L_prior directly supervises
unified prior_mask — no per-probe max-pool aggregation.

### Vendored third-party code

MobileSAM is vendored under `thirdparty/MobileSAM/` and injected via `sys.path` at runtime by `adasam/backbone/mobile_sam.py:_ensure_mobile_sam_on_path()`. It is NOT declared as a pip package. The weights file (`weights/mobile_sam.pt`, ~40 MB) is gitignored. Tests that depend on weights auto-skip when the file is absent.

### Evaluation (FSS Benchmark Protocol)

Core metrics computed by `tools/adasam/eval.py`:
1. **mIoU** — per-class IoU averaged over all visible classes
2. **FB-IoU** — foreground-background IoU (FSS standard, FG=union of all visible classes)
3. **Per-class IoU** — with GT tile counts and support tile counts
4. **3-fold CV** — fold 0/1/2 with mean
5. **Multi-seed** — Mean±Std for reproducibility

Support cache is fixed per evaluation run (FSS standard protocol).

### Configuration

- Stage 1: `configs/stage1.yaml`
- Stage 2: `configs/base.yaml` and `configs/isaid_5i.yaml`
CLI arguments override YAML fields. The checkpoint's embedded config is the default source during evaluation.

### Dataset

- **iSAID-5i**: 15-class semantic segmentation (PNG annotations), 256² tiles, 3-fold base/novel splits. FSS Benchmark protocol.
- **NEU-SEG**: 6-class industrial defect segmentation, 480×640, 35 images.

`EpisodeSampler` enforces scene-disjoint sampling (support and query from different source images) with a `min_tiles` filter.

### Logging

Structured logging system (`adasam/logging/`) — all observable values go through `get_logger(name)` with named backends (ConsoleBackend, FileBackend). No bare `print()`. File output is JSONL to `runs/` (gitignored). Backend write failures are silently caught to never crash training.

### Experiment tracking

`EXPERIMENT_MANIFEST.md` is the traceability registry. Every paper number binds: ID → Protocol → Manifest → Seed → Checkpoint → Commit.
