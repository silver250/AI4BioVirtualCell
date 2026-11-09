import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
