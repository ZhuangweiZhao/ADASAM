# LoveDA Benchmark Comparison

Last updated: 2026-08-16

## Comparison boundary

The literature values below were supplied by the user. Their original citations,
data splits, image sizes, training recipes, pretraining, test-time augmentation,
and evaluation implementations have not yet been verified. They are external
context, not a strict same-protocol ranking or verified SOTA evidence.

AdaSAM results use: official Train with a fixed 20% internal validation split;
official Val as test; 512 x 512; batch 8; 100% labels; seed 42; class-balanced CE;
cosine scheduling; gradient clipping. They are single-seed screening results.

## User-supplied literature values (%)

| Method | Background | Building | Road | Water | Barren | Forest | Agriculture | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net R50 | 43.06 | 52.74 | 52.78 | 73.08 | 10.33 | 43.05 | 59.87 | 47.84 |
| HRNet-W32 | 44.61 | 55.34 | 57.42 | 73.96 | 11.07 | 45.25 | 60.88 | 49.79 |
| UNetFormer | 44.70 | 58.80 | 54.90 | 79.60 | 20.10 | 46.00 | 62.50 | 52.40 |
| Hi-ResNet | 46.70 | 58.30 | 55.90 | 80.10 | 17.00 | 46.70 | 62.70 | 52.50 |
| AerialFormer-B | 47.80 | 60.70 | 59.30 | 81.50 | 17.90 | 47.90 | 64.00 | 54.10 |
| MTP | 46.80 | 62.60 | 58.96 | 82.25 | 17.49 | 47.63 | 63.44 | 54.17 |
| SegFormer-B5 | 46.54 | 57.46 | 58.91 | 80.09 | 27.89 | 46.14 | 61.00 | 54.01 |
| U-Net MaxViT-S | 48.59 | 60.47 | 63.40 | 81.17 | 27.02 | 48.10 | 64.40 | 56.16 |

Status: citation and protocol verification pending.

## AdaSAM unified 100% screening results

| Model | mIoU (%) | Boundary F1 (%) | Small IoU (%) | Trainable parameters |
|---|---:|---:|---:|---:|
| MobileSAM full finetune | **52.35** | 26.32 | 13.49 | 6.37M |
| DeepLabV3+-R50 | 51.10 | 25.79 | **14.91** | 26.68M |
| SegFormer-B0 | 49.45 | **27.36** | 13.38 | 3.72M |
| Frozen MobileSAM + Sum | 47.69 | 24.18 | 13.53 | **0.303M** |

## MobileSAM full-finetune comparison

| Class/metric | MobileSAM (%) | Best supplied (%) | Difference (points) |
|---|---:|---:|---:|
| Background | **53.69** | 48.59 | **+5.10** |
| Building | 60.15 | 62.60 | -2.45 |
| Road | 56.53 | 63.40 | -6.87 |
| Water | 69.63 | 82.25 | -12.62 |
| Barren | **31.24** | 27.89 | **+3.35** |
| Forest | 36.07 | 48.10 | -12.03 |
| Agriculture | 59.10 | 64.40 | -5.30 |
| mIoU | 52.35 | 56.16 | -3.81 |

MobileSAM full finetune is +4.51 points over U-Net R50, +2.56 over HRNet-W32,
-0.05 from UNetFormer, -0.15 from Hi-ResNet, -1.66 from SegFormer-B5, -1.75
from AerialFormer-B, -1.82 from MTP, and -3.81 from U-Net MaxViT-S.

## Interpretation

- The 52.35% result is competitive with the supplied UNetFormer and Hi-ResNet
  values, but is not the highest reported result in this table.
- MobileSAM is strongest on background and barren land. The largest gaps are
  water and forest, identifying vegetation/water discrimination as a bottleneck.
- Frozen MobileSAM is a parameter-efficiency trade-off: only 0.303M trainable
  parameters, but 4.66 mIoU points below full finetuning internally.
- Cross-source deltas are not paper claims until citation and protocol checks are
  complete. Final AdaSAM claims also require seeds 42, 123, and 456.

