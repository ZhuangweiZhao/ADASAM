# Exemplar Learning Dossier

## Exemplar inventory

| Pattern ID | Paper pattern | Why relevant | AdaSAM use |
|---|---|---|---|
| E1 | Controlled low-label benchmark | Converts many runs into stable empirical findings | Make annotation budget and seed pairing the organizing axes |
| E2 | Parameter-efficient adaptation study | Separates trainable capacity from accuracy | Compare U-Net, frozen MobileSAM and DAPG under matched subsets |
| E3 | Negative-results-aware ablation | Reports mechanisms that do not survive profiling | Preserve the LoveDA sparse-retention null result |
| S1 | SAM adaptation method paper | Requires component necessity and full efficiency accounting | Use only if fixed-protocol DAPG/CRHC evidence is completed |
| S2 | Remote-sensing segmentation benchmark | Requires verified split and literature comparability | Treat LoveDA external values as context, not ranking |
| S3 | Multi-seed robustness study | Makes seed-level paired effects central | Report effect size and variance, not best seed |

## Structural patterns

The strongest structure for current evidence is: problem and protocol gap; controlled
matrix; annotation-budget scaling; augmentation-versus-adaptation attribution;
cross-dataset survival tests; negative evidence; limitations and preregistered next
experiments. A method-first structure would currently force unsupported claims.

## Rhetorical patterns

Use claim-evidence-boundary triples. State exact budgets and paired effects. Describe
null results as design information. Avoid `real-time`, `10x faster`, `SOTA`,
`significantly better`, and `matches full supervision` until the missing measurements
and statistical tests exist.

## Language patterns

Prefer `under the evaluated protocol`, `consistently across three seeds`, `screening
evidence suggests`, and `does not establish`. Reserve causal language for controlled
comparisons in which only one factor changes.
