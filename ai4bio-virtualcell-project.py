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


@app.cell
def _(perts_df):
    print(perts_df.iloc[:, 2])
    return


@app.cell
def _():
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
    small_val_adata = val_adata
    return (small_val_adata,)


@app.cell
def _(np, random, small_val_adata):
    # Create mask that identifies all control cells, and then get all of their indices into control_indices.
    is_control = (small_val_adata.obs["target_gene"] == "non-targeting").to_numpy()
    control_indices = np.flatnonzero(is_control)

    # Start with all the indices of the adata object, which is 0 - (n-1). 
    # Then, for all the indices not in control, pick a random control index
    val_preds_indices = np.arange(small_val_adata.n_obs)
    val_preds_indices[~is_control] = [random.choice(control_indices) for x in val_preds_indices[~is_control]]

    # Make a new adata object with a new X indexed by the control-sampled predictions.
    val_pred_nochange = small_val_adata.copy()
    val_pred_nochange.X = small_val_adata.X[val_preds_indices]
    val_pred_nochange
    return (val_pred_nochange,)


@app.cell
def calc_mae(small_val_adata, val_pred_nochange):
    from cell_eval2 import compute_metrics, aggregate_metrics

    metrics_df = compute_metrics(val_pred_nochange, small_val_adata,
                                metrics=["expr_mae", "pds_l1"],
                                pert_col="target_gene", control="non-targeting")
    metrics_df
    return aggregate_metrics, metrics_df


@app.cell
def _(aggregate_metrics, metrics_df):
    print(aggregate_metrics(metrics_df)) 
    return


if __name__ == "__main__":
    app.run()
