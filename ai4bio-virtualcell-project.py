# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anndata>=0.13.4",
#     "cell-eval2>=0.22.0",
#     "gcsfs>=2026.10.1",
#     "marimo>=0.25.2",
#     "numpy==2.5.4",
#     "pandas>=3.1.1",
#     "pathlib>=1.0.1",
#     "pyarrow>=26.0.1",
#     "scanpy>=1.12.5",
#     "truststore>=0.10.5",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import pyarrow as pa
    import pandas as pd
    import anndata as ad
    import gcsfs
    import scanpy as sc
    from pathlib import Path
    import random
    import numpy as np

    return Path, ad, np, pd, random


@app.cell
def _(Path, ad):
    import marimo as mo

    adata_path = Path(mo.notebook_dir()) / "data/train/adata_Training.h5ad"
    adata = ad.read_h5ad(adata_path)
    return adata, mo


@app.cell
def _(adata):
    counts_small_df = adata[:1000, :1000].to_df()
    counts_small_df.head()
    return


@app.cell
def _(adata):
    # print(counts_data.obs.columns)
    # print(counts_small_df.columns)
    adata.obs.columns                                  # every column that holds labels

    print("All labels under target gene")
    print()
    print(adata.obs["target_gene"].unique())    # all perturbation labels

    print("All counts of the unique values")
    print(adata.obs["target_gene"].value_counts())       # how many cells each label has

    print("Gene metadata")
    print(adata.var)
    return


@app.cell
def _(Path, mo, pd):
    perts_path = Path(mo.notebook_dir()) / "data/train/pert_counts_Training.csv"
    perts_df = pd.read_csv(perts_path)
    return (perts_df,)


@app.cell
def _(perts_df):
    perts_df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **No-Change Baseline**

    To predict the gene count vectors of perturbed cells, the first method we try is the no-change baseline. For each cell in the perturbed group, this method samples a random cell in the control group and predicts the sample's gene counts. This method makes the assumption that each perturbation will have no effect. This method only uses the validation set, not the training set.
    """)
    return


@app.cell
def _(Path, ad, mo):

    val_adata_path = Path(mo.notebook_dir()) / "data/validation/adata_Validation.h5ad"
    val_adata = ad.read_h5ad(val_adata_path)
    # val_adata = val_adata[:5000]
    return (val_adata,)


@app.cell
def _(np, random, val_adata):
    # Create mask that identifies all control cells, and then get all of their indices into control_indices.
    is_control = (val_adata.obs["target_gene"] == "non-targeting").to_numpy()
    control_indices = np.flatnonzero(is_control)

    # Start with all the indices of the adata object, which is 0 - (n-1). 
    # Then, for all the indices not in control, pick a random control index
    val_preds_indices = np.arange(val_adata.n_obs)
    val_preds_indices[~is_control] = [random.choice(control_indices) for x in val_preds_indices[~is_control]]

    # Make a new adata object with a new X indexed by the control-sampled predictions.
    val_pred_nochange = val_adata.copy()
    val_pred_nochange.X = val_adata.X[val_preds_indices]
    val_pred_nochange
    return is_control, val_pred_nochange


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Then, we evaluate this prediction method using the metrics computation API provided in the cell-eval2 repository from the competition.
    """)
    return


@app.cell
def calc_mae(val_adata, val_pred_nochange):
    from cell_eval2 import compute_metrics, aggregate_metrics

    no_change_metrics_df = compute_metrics(val_pred_nochange, val_adata,
                                metrics=["expr_mae", "pds_l1"],
                                pert_col="target_gene", control="non-targeting")
    no_change_metrics_df
    return aggregate_metrics, compute_metrics, no_change_metrics_df


@app.cell
def _(aggregate_metrics, no_change_metrics_df):
    print(aggregate_metrics(no_change_metrics_df)) 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##**Mean Baseline**

    Under this baseline model, instead of selecting a random control cell for each perturbed cell, we will find the "mean" control cell, or the theoretically average sample in the control group, and predict that same mean for each of the perturbed cells.
    """)
    return


@app.cell
def _(val_adata):
    small_val_adata = val_adata[:500]
    return (small_val_adata,)


@app.cell
def _(np, small_val_adata):

    # Get the subset of gene count rows that are controls and find the mean row, where each element is the mean of its column.
    control_group = small_val_adata[small_val_adata.obs["target_gene"] == "non-targeting"].X
    control_mean = np.mean(control_group, axis=0)
    return (control_mean,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As in the no-change baseline, we map each perturbed cell to its prediction, which in this case is the same theoretical control mean.
    """)
    return


@app.cell
def _(control_mean, is_control, small_val_adata):
    # Make a new adata object with a new X indexed by the control-sampled predictions.
    val_pred_mean = small_val_adata.copy()

    # Replace all perturbed rows with the control mean.
    val_pred_mean.X[~is_control[:500]] = control_mean
    return (val_pred_mean,)


@app.cell
def _(aggregate_metrics, compute_metrics, small_val_adata, val_pred_mean):
    mean_metrics_df = compute_metrics(val_pred_mean, small_val_adata,
                                      metrics=["expr_mae", "pds_l1"],
                                      pert_col="target_gene", control="non-targeting",
                                      allow_fractional_counts=True)
    mean_metrics_df
    print(aggregate_metrics(mean_metrics_df)) 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Network KNN comparison

    The [KNN codebook](knn_workflow/notebooks/model_codebook.py) explains the
    models and displays the full-validation expression and discrimination
    comparison. The [workflow README](knn_workflow/README.md) explains how to
    reproduce it. The codebook includes the full-validation DE comparison.
    """)
    return


if __name__ == "__main__":
    app.run()
