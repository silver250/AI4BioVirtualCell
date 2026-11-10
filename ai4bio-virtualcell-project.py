# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anndata>=0.13.4",
#     "gcsfs>=2026.10.1",
#     "marimo>=0.25.2",
#     "pandas>=3.1.1",
#     "pyarrow>=26.0.1",
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
    import anndata
    import gcsfs
    import truststore
    truststore.inject_into_ssl()
    return (pd,)


@app.cell
def _(pd):
    import marimo as mo
    meta = pd.read_parquet("sample_metadata.parquet")  # same folder as notebook.py
    meta
    return


if __name__ == "__main__":
    app.run()
