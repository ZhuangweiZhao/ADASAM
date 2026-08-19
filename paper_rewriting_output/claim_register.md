# Claim Register

| Claim ID | Draft claim | Evidence | Allowed wording | Status |
|---|---|---|---|---|
| CL01 | The study provides a matched low-label matrix. | E01 | `we evaluate` / `the matrix contains` | allowed |
| CL02 | Basic augmentation yields larger low-label gains than observed DAPG-v2 gains. | E02–E03 | `in this matrix`, `observed` | allowed with scope |
| CL03 | DAPG-v2 is uniformly better than frozen MobileSAM. | E03 | none | reject |
| CL04 | Frozen MobileSAM is competitive under some full-supervision screening settings. | E05 | `screening result` | allowed with scope |
| CL05 | Adaptive retention improves Vaihingen mIoU over random across three seeds. | E06 | `screening evidence suggests` | auxiliary |
| CL06 | Sparse retention accelerates end-to-end inference. | E07 | none | reject |
| CL07 | Results generalize to all remote-sensing domains. | E05–E06 | none | reject |
| CL08 | The paper establishes a reproducible prioritization protocol. | E01–E08 | `we propose an evidence-based protocol` | conditional on fixed reruns |
| CL09 | Results are paper-ready and fully reproducible. | E08 | none | reject until gaps close |
