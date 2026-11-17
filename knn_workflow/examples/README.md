# Validation summary

`all_validation_cells.csv` summarizes exploratory expression MAE and PDS on all 98,927 validation cells, 50 targets and 18,080 genes. Each model uses the same 500 sampled training controls to generate predicted cells. A model's group averages are compared with the average expression of every real validation cell in that group.

Differential-expression (DE) metrics require individual-cell variation and are being computed separately; they will be added later. Official test responses were not used. The validation targets have been reused during exploration; this is not an independent final test.
