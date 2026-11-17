# Validation summary

`all_validation_cells.csv` summarizes exploratory expression MAE and PDS on all 98,927 validation cells, 50 targets and 18,080 genes. Each model uses the same 500 sampled training controls to generate predicted cells. A model's group averages are compared with the average expression of every real validation cell in that group.

`full_validation_de_summary.csv` and `full_validation_de_per_target.csv` contain the full-cell differential-expression (DE) results for the same targets and prediction samples. All real validation cells and 38,176 real controls were used, with 500 predicted cells per target. The audit is recorded in `full_validation_de_audit.json`. These are exploratory validation results, not official test results. The validation targets have been reused, and DE depends on the predicted-cell construction.
