import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    return Path, mo, pd, plt


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

    | Result name | Plain-language model |
    | --- | --- |
    | `no_change` | Predict no change in any gene. |
    | `mean_effect` | Average the responses of all training targets. |
    | `direct_equal` | Average up to five direct STRING neighbors equally. |
    | `direct_score` | Use the same direct neighbors, but weight each response by its STRING score. |
    | `evidence_equal` | Require experimental or curated support, then average direct neighbors equally. |
    | `dsd_equal` | Average the five closest training targets according to network walk patterns. |
    | `dsd_equal_k10` | Use the same DSD method with up to ten training targets. |

    The one-hop models use all-channel combined STRING scores at or above 0.4.
    In `direct_score`, a neighbor with a larger combined score gets a larger
    share of the average. Evidence filtering adds a support requirement; it
    still ranks by the combined score. DSD uses the full graph, including
    intermediate proteins, to compare six-step walk patterns.

    A STRING score measures confidence in an association, not the size of a
    knockdown response. If a network model has no eligible training neighbor,
    it uses the mean-effect baseline.
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
    **PDS (perturbation discrimination score):** higher means the prediction matches its own
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
    results = _expression.merge(_de, on="method", validate="one_to_one")
    mo.vstack([mo.md("### Expression, discrimination, and DE overlap"), results])
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How the comparison was run

    1. **Training:** all training cells were normalized to 10,000 counts per
       cell and log-transformed. We estimated each training target's average
       response relative to non-targeting controls.
    2. **Predictions:** each model predicted every validation target using only
       those training responses and allowed STRING network information. For
       every model and target, we used the same 500 sampled training controls,
       added the predicted response, and clipped negative values to zero.
    3. **Expression and PDS:** we streamed every real validation cell into its
       group's average, then compared those averages with the predicted groups.
       These two metrics depend on the group averages in this configuration.
    4. **DE overlap:** a separate memory-bounded CPU calculation compared the
       individual real and predicted cells with all 38,176 real controls. It
       used a tie-corrected Wilcoxon test and per-target multiple-test
       adjustment. In this `cell-eval2` 0.22.0 Scanpy/log-normalized setup,
       the configured CPM gene filter is skipped.

    The DE calculation was checked against Scanpy and audited across all 50
    targets and seven models. This checks the arithmetic, not whether the
    predicted cells have realistic variation. Earlier exploratory checks used
    20 validation targets to compare neighbor counts and one shuffled-label
    control; they are not the full-validation results shown here. Validation
    has been reused for exploration; official test responses remain untouched.
    """)
    return


@app.cell
def _(plt, results):
    # Show the three scores separately because they have different scales and
    # directions. Keep the same model order in every panel for easy comparison.
    _labels = {
        "no_change": "No change",
        "mean_effect": "Mean effect",
        "direct_equal": "One-hop equal",
        "direct_score": "One-hop score-weighted",
        "evidence_equal": "Evidence-filtered equal",
        "dsd_equal": "DSD, k=5",
        "dsd_equal_k10": "DSD, k=10",
    }
    _order = list(_labels)
    _plot_data = results.set_index("method").loc[_order]
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for _ax, _column, _title, _color in zip(
        _axes,
        ["expr_mae", "pds_l1", "de_wilcoxon_overlap"],
        ["Expression MAE (lower is better)", "PDS (higher is better)", "DE overlap (higher is better)"],
        ["#397a9b", "#528b62", "#ac7b39"],
    ):
        _ax.barh([_labels[_name] for _name in _order], _plot_data[_column], color=_color)
        _ax.set_title(_title, fontsize=10)
        _ax.grid(axis="x", alpha=0.2)
        _ax.set_axisbelow(True)
    _axes[0].invert_yaxis()
    _fig.tight_layout()
    _fig
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
