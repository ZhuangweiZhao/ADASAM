# Final Artifact Manifest

| Artifact | Category | Status | Notes |
|---|---|---|---|
| `paper_spine_config.json` | required | present | Conference, build-from-materials workflow |
| `confirmed_contribution.md` | required | present | Contribution A, passed contribution check |
| `confirmed_motivation.md` | required | present | Motivation A, passed motivation gate |
| `source_inventory.md` | required | present | Material inventory |
| `evidence_bank.md` | required | present | Eight evidence groups with boundaries |
| `figure_asset_map.md` | required | present | Seven planned/candidate figures |
| `claim_register.md` | required | present | Allowed and rejected claims |
| `section_blueprints.md` | required | present | Conference-paper blueprint |
| `writing_rationale_matrix.md` | required | present | 18 claim-bearing units |
| `final_paper/main.tex` | required | present | Initial English manuscript draft |
| `final_paper/references.bib` | required | present | Bibliography keys used by LaTeX |
| `final_paper/main.pdf` | required | blocked | TeX engine unavailable |
| `final_paper/paper.pdf` | required | blocked | TeX engine unavailable |
| `final_paper/paper.docx` | required | blocked | Pandoc unavailable |
| `word_report.md` | required | blocked | Word file unavailable |

The workflow is not complete until the blocked PDF and Word artifacts are built
and checked on an environment containing `pdflatex`, `bibtex` and `pandoc`.
