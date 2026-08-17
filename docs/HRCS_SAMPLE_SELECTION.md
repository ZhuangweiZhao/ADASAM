# LoveDA HRCS Survival Experiment

## Question

Does coverage of frozen MobileSAM's hierarchical representation space improve a
5% annotation budget independently of CRHC?

## Fixed Protocol

- Dataset: LoveDA
- Validation split: 20%, seed 42
- Selection pool: the remaining 2,018 training images only
- Annotation budget: 5%, 101 images
- Adaptation: MobileSAM LoRA rank 4, `qkv + proj`
- Fusion: Dense Sum
- Input and SAM size: 512
- Training seed: 42

Only the selected 101 training images may differ between runs.

## Generate Selection Manifests

```bash
cd /root/ADASAM

python tools/select_loveda_coreset.py \
  --data-root /root/autodl-tmp/LoveDA \
  --checkpoint /root/ADASAM/weights/mobile_sam.pt \
  --output-dir /root/autodl-tmp/runs/loveda_hrcs_manifests \
  --label-ratio 5 \
  --selection-seed 42 \
  --validation-seed 42 \
  --sam-image-size 512 \
  --pca-dim 64 \
  --batch-size 8 \
  --num-workers 8 \
  --device cuda
```

This produces:

```text
random_ratio5.json
embedding_kcenter_ratio5.json
hrcs_ratio5.json
selection_statistics.json
```

The selector opens RGB images only. It never reads LoveDA masks.

## Run All Three Experiments

```bash
python tools/run_loveda_selection_survival.py \
  --data-root /root/autodl-tmp/LoveDA \
  --checkpoint /root/ADASAM/weights/mobile_sam.pt \
  --manifest-dir /root/autodl-tmp/runs/loveda_hrcs_manifests \
  --output-dir /root/autodl-tmp/runs/loveda_hrcs_survival \
  --epochs 100 \
  --batch-size 8 \
  --num-workers 8 \
  --device cuda
```

Use `--methods embedding_kcenter hrcs` to reuse the existing random baseline,
although rerunning all three gives the cleanest matched comparison.

## Decision Rule

Compare the best of Embedding k-center and HRCS against Random LoRA + Sum:

- gain below 0.3 mIoU: stop the selection direction;
- gain from 0.3 to 0.8: retain as an auxiliary contribution;
- gain from 0.8 to 1.5: continue as a second core contribution;
- gain at least 1.5: proceed to routing-aware 5% to 10% expansion.

Both `mean_nearest_distance` and `max_nearest_distance` are distances, so lower
values indicate better representation-space coverage.
