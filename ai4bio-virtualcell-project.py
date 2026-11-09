import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Predicting unseen CRISPRi responses

        This is the shared marimo entry point for our VCC 2025 AI for Biology
        project. We predict expression after knocking down a target gene whose
        perturbation response was not used for training.

        **Main comparison:** no change, mean effect, STRING nearest neighbors,
        and a small neural network. See `docs/PROJECT_PLAN.md` for definitions,
        split rules, and evaluation.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## First data checkpoint

        1. Download the public VCC 2025 data into `data/raw/`.
        2. Confirm the count matrix, control label, target gene labels, and
           target-level train, validation, and test groups.
        3. Build a small no-change prediction and verify its format with
           `cell_eval2` before adding models.

        Keep reusable code in `src/` once it exists. Keep large data and
        prediction files outside Git.
        """
    )
    return


if __name__ == "__main__":
    app.run()
