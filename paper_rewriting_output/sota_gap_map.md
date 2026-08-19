# SOTA Gap Map

| Candidate contribution | What prior work broadly does | User evidence | Real gap | Claim strength | Risk |
|---|---|---|---|---|---|
| Controlled empirical law for low-label adaptation | Compares architectures or PEFT methods, often under individual budgets | 108 runs across 3 models, 2 augmentation modes, 6 budgets and 3 seeds | Factor attribution across annotation budget, adaptation and augmentation | Moderate, after protocol correction | Legacy split and incomplete provenance |
| AdaSAM/DAPG method superiority | Uses prompts, prototypes, adapters and lightweight decoders | Small and inconsistent paired DAPG gains | No stable evidence that the proposed module is the decisive factor | Weak | Novelty collapse and non-significant gains |
| Adaptive hierarchical retention | Retains/selects multi-scale features | Consistent +0.9-1.2 mIoU and Small-IoU gain over random on Vaihingen | Selection policy may preserve small-object detail | Moderate auxiliary claim | No efficiency proof; narrow dataset |
| Semantic representation budget | Uses multi-level SAM features | LoveDA K=3 > K=2 > K=1 at 5/10/20% | How representation budget scales with annotation budget | Preliminary | Mostly single seed |
| Sparse adaptation acceleration | Prunes feature locations or layers | Retention sweep has flat FPS around 88-89 | Framework sparse operators may not translate sparsity to speed | Negative finding only | Cannot support acceleration claim |

## Gap summary

The defensible gap is not that SAM lacks another adapter. It is that current project
evidence does not yet distinguish architectural adaptation gains from annotation-
preserving augmentation and protocol effects across label budgets. The replicated
matrix can address this as an empirical study; the method claim needs new fixed-
protocol experiments.
