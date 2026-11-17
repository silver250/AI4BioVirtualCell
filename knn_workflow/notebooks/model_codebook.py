import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    import pandas as pd
    return Path, mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Predicting unseen knockdown responses

    We ask whether a gene's network neighbors can help predict its expression
    response when that target was not in the training set. We estimate each
    known response by subtracting average control expression from average
    expression after a knockdown, one measured gene at a time.

    Validation responses are used for comparison. The official test responses
    remain held out. The models and results here are exploratory.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The models

    | Method | How the response is predicted |
    | --- | --- |
    | No change | Predict no change in any gene. |
    | Mean effect | Average the responses of all training targets. |
    | One-hop equal | Average up to five directly connected training targets equally. |
    | One-hop weighted | Use the same neighbors but give higher STRING scores more weight. |
    | Evidence-filtered equal | Require experimental or curated support before selecting direct neighbors. |
    | DSD | Compare network walk patterns, then average up to five or ten closest training targets. |

    STRING scores describe confidence in a functional association. They are
    not the measured strength of a knockdown response. When a network method
    has no eligible training neighbor, it uses the mean-effect baseline.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Validation results

    Every real validation cell was used: 98,927 cells, including controls,
    across 50 held-out targets and 18,080 measured genes. Each prediction uses
    the same 500 sampled training controls, shifted by the predicted response.
    Negative values are clipped to zero.

    **Expression MAE:** lower means the mean expression profile is closer.
    **PDS:** higher means the predicted response is better at matching its own
    target rather than other targets. **DE overlap:** higher means more overlap
    between the real and predicted differentially expressed genes. All three
    metrics depend on the target panel and the predicted-cell assumptions.
    """)
    return


@app.cell
def _(Path, mo, pd):
    _examples = Path(mo.notebook_dir()).parent / "examples"
    _expression = pd.read_csv(_examples / "all_validation_cells.csv")
    _de = pd.read_csv(_examples / "full_validation_de_summary.csv")
    _results = _expression.merge(_de, on="method", validate="one_to_one")
    mo.vstack([mo.md("### Expression, discrimination, and DE overlap"), _results])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Scanpy warning

    - **When:** A gene has the same value in every cell being compared, often zero.
    - **Why:** A statistical test cannot calculate a meaningful difference without variation.
    - **What happens:** Scanpy assigns that gene a neutral p-value of 1; it is not called differentially expressed.

    This warning is separate from whether our predicted cells capture realistic
    cell-to-cell variation. See `docs/ASSUMPTIONS.md` for the evaluation limits.
    """)
    return


if __name__ == "__main__":
    app.run()
