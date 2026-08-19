# Evidence Bank

| Evidence ID | Claim supported | Result | Strength | Source |
|---|---|---|---|---|
| E01 | Replicated budget matrix exists | 108 NEU-Seg runs: 3 models x 2 augmentation modes x 6 budgets x 3 seeds | screening | `EXPERIMENT_MANIFEST.md` |
| E02 | Basic augmentation is a strong low-label factor | At 1%, gain is +15.683 U-Net, +6.115 MobileSAM and +6.919 DAPG-v2 mIoU points | screening | `table3_basic_augmentation_gain_points.csv` |
| E03 | DAPG-v2 is not uniformly superior | Paired gains from 1% to 100% are -0.632, +0.976, +0.489, +0.011, +0.111 and +0.416 points | screening | `table4_dapg_paired_gain_points.csv` |
| E04 | Accuracy scales with budget | Basic-augmentation DAPG-v2 rises from 0.589 at 1% to 0.808 at 100% | screening | `table1_basic_miou_fg.csv` |
| E05 | Frozen MobileSAM is competitive in LoveDA full supervision | 47.69 mIoU with 0.303M trainable parameters versus 52.35 full finetune | single-seed screening | `docs/LOVEDA_BENCHMARK_COMPARISON.md` |
| E06 | Adaptive retention has a reproducible auxiliary signal | Vaihingen adaptive beats random across three seeds by about 0.9–1.2 mIoU | screening | `EXPERIMENT_MANIFEST.md` |
| E07 | Sparse retention does not establish speedup | LoveDA magnitude retention has flat FPS around 88–89 | screening/null | `EXPERIMENT_MANIFEST.md` |
| E08 | Evidence is incomplete | Checkpoints, provenance and fixed efficiency records are missing for headline bundles | blocking limitation | `EXPERIMENT_MANIFEST.md` |
