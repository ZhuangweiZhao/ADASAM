# Source Map

| Source group | Authoritative files | Role | Claim boundary |
|---|---|---|---|
| Current experiment registry | `EXPERIMENT_MANIFEST.md`, `runs/run_index.csv` | Determines which runs exist and their evidence status | All registered groups are currently `screening`, not paper-ready |
| NEU-Seg replicated matrix | `analysis/multiseed_tables/table1_basic_miou_fg.csv` through `table5_full_supervision_retention_percent.csv` | Three-model, six-budget, three-seed accuracy evidence | Legacy split protocol; timing table is inadmissible |
| Current method intent | `METHOD_DESIGN.md`, `docs/TECHNICAL_DOCUMENTATION.md` | Defines AdaSAM components and intended accuracy-efficiency claim | Design statements are hypotheses unless linked to controlled results |
| LoveDA evidence | `docs/LOVEDA_BENCHMARK_COMPARISON.md`, cloud results under `runs/current/cloud/` | Full-supervision context, semantic-budget and layer studies | Mostly single-seed; external literature table is unverified |
| Vaihingen evidence | `EXPERIMENT_MANIFEST.md`, `runs/current/vaihingen/adaptive_survival/` | Three-seed adaptive-retention survival evidence | Small gain; no validated efficiency claim |
| Negative evidence | `docs/not_yet_proven.md`, `docs/research_log.md` | Prevents novelty and causal overclaims | Must remain visible in abstract, results and limitations |
| Literature context | `docs/SAM_Remote_Sensing_Survey.md`, `docs/literature_landscape.md` | Local field map and candidate related work | Bibliographic entries require later citation verification |

## Evidence precedence

Machine-readable current metrics take precedence over narrative summaries. The
experiment manifest controls admissibility. Historical, development, sanity and
diagnostic runs may explain decisions but cannot support headline claims.
