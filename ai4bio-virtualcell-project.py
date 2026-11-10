# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anndata>=0.13.4",
#     "gcsfs>=2026.10.1",
#     "marimo>=0.25.2",
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

    return Path, ad


@app.cell
def _(Path, ad):
    import marimo as mo

    path = Path(mo.notebook_dir()) / "data/raw/vcc_2025/train/adata_Training.h5ad"
    adata = ad.read_h5ad(path)  
    return (adata,)


@app.cell
def _(adata):
    expr = adata[:1000, :1000].to_df()   
    expr.head()
    return


if __name__ == "__main__":
    app.run()
