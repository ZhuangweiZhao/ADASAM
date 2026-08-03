# Multiseed table notes

Tables 1-5 summarize the legacy split protocol and remain useful as exploratory results.

`table6_training_efficiency_basic.csv` must not be used as a paper-ready efficiency comparison. In the legacy runs, U-Net epoch time included validation while MobileSAM epoch time covered only training. The timing boundary is corrected for all future fixed-protocol runs. Regenerate Table 6 from the fixed-protocol summary before publication.

Fixed protocol definition:

- validation seed: 42
- fixed validation set: 726 images
- remaining training pool: 2,904 images
- label ratios are applied to the training pool
- training subsets are nested for a given experiment seed
- validation and test augmentation: none
