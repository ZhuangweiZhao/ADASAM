# EXPERIMENT_MANIFEST — AdaSAM

Single traceability registry. Each paper number binds:
**ID → Protocol → Manifest → Seed → Checkpoint → Commit**. A number may enter a paper table only
when its row is `Protocol=V3` with a frozen Manifest, a Checkpoint, and a Commit hash.

## Fixed constants

| Field | Value |
|---|---|
| Protocol | **V3** (COCO AP, one-to-one instance matching, no union masks) |
| Evaluator | `tools/evaluate.py` (single sanctioned entry point) |
| Metrics | `adasam/metrics/{instance_match,coco_eval}.py` (frozen core, ported verbatim) |
| Manifest | `<data_root>/evaluation_manifest_val.json` (frozen query set) |
| Standard seeds | `42, 123, 456` |
| Product | `instance_metrics.json` |

## Registry

| ID | Config | Decoder / Proto | K | Seeds | Manifest | Checkpoint | Commit | V3 Status |
|----|--------|-----------------|---|-------|----------|------------|--------|-----------|
| AS-00 | MobileSAM zero-shot (everything-mode, class-agnostic) | — | 0 | 42 | ⏳ | — | — | ⏳ pending |
| AS-01 | MobileSAM + prototype-prompt few-shot | prompt / p-embed | 5 | 42/123/456 | ⏳ | — | — | ⏳ pending |

Status legend: ✅ done · ⏳ pending · — n/a.

> Rows are backfilled at M8 (baseline reproduce) with real checkpoints + `git rev-parse --short HEAD`.
