# Research Dossier

## Target scene

The configured target is a machine-learning conference paper. A defensible paper
therefore needs one falsifiable contribution, matched baselines, repeated trials,
explicit protocol control, and a result structure that separates positive,
negative and efficiency findings.

## Review criteria

1. **Novelty:** more than combining MobileSAM, LoRA, prototypes and a decoder.
2. **Technical soundness:** nested label subsets, frozen validation/test data,
   matched training budgets and at least three seeds for headline comparisons.
3. **Evidence completeness:** checkpoints, resolved configurations, commit and
   environment provenance, uncertainty and efficiency boundaries.
4. **Significance:** either a clear Pareto improvement or a reusable empirical
   insight about label-efficient adaptation.
5. **Reproducibility:** exact annotation counts, manifests, trainable parameters,
   hardware and timing protocol.

## Accepted-paper patterns applicable here

- A method paper organizes Results around a claim ladder: accuracy, component
  necessity, robustness/generalization, then cost.
- An empirical paper can be publishable when the protocol is unusually controlled,
  the matrix is broad enough to expose stable laws, and negative results change
  how practitioners allocate effort.
- Small average gains require paired seed-level analysis and should not be sold as
  universal superiority.

## Constraints for AdaSAM

- The NEU-Seg 108-run matrix is the broadest replicated evidence, but its split is
  marked legacy and its timing boundary is inconsistent.
- DAPG-v2 gains over frozen MobileSAM are modest: with basic augmentation the
  paired mean gain is -0.63, +0.98, +0.49, +0.01, +0.11 and +0.42 mIoU points from
  1% to 100%, respectively.
- Basic augmentation is a much larger factor, adding 6.1-6.9 points at 1% for the
  MobileSAM/DAPG models and 15.7 points for U-Net in the legacy matrix.
- Vaihingen adaptive retention beats random across three seeds, but only by about
  0.9-1.2 mIoU points and lacks a fixed efficiency record.
- LoveDA semantic-budget monotonicity is promising but mainly single-seed; magnitude
  retention produced essentially flat accuracy and FPS and therefore falsifies the
  current sparse-speed narrative.

## Research conclusion

The current evidence supports an empirical contribution about what actually drives
label-efficient segmentation performance more strongly than it supports a new-method
superiority claim. A method-centric AdaSAM contribution remains possible only after
the fixed-protocol evidence gaps are closed.
