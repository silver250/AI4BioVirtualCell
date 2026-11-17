# Cell Mates: predicting unseen knockdown responses

We ask whether a gene network helps predict a knockdown response for a target absent from training. This is a **research draft**: we have compared validation targets, but have not frozen a winning model or run a final test evaluation.

## Start here

- [Model codebook](notebooks/model_codebook.py): first-principles walkthrough and small saved result tables. Run `uv run marimo edit notebooks/model_codebook.py`.
- [Assumptions and evaluation](docs/ASSUMPTIONS.md): what the models and metrics mean, and the remaining decisions.
- [Data instructions](data/README.md): keep large inputs and cell IDs outside Git.
- [Example results](examples/README.md): what the saved aggregate tables show.

## Where the code is

| File | Purpose |
| --- | --- |
| `src/cellmates/baselines.py` | No-change and mean-effect predictions. |
| `src/cellmates/direct.py` | One-hop equal and score-weighted KNN. |
| `src/cellmates/dsd.py` | Finite-step diffusion-state-distance KNN. |
| `src/cellmates/preprocessing.py` | Training cell means and response vectors. |
| `src/cellmates/cells.py` | Experimental control-based predicted cell populations. |
| `src/cellmates/neighbor_cells.py` | Experimental transfer of observed training-neighbor cells. |
| `src/cellmates/evaluation.py` | Common `cell_eval2` configuration. |
| `scripts/` | Explicit training preprocessing, network preparation and validation runs. |
| `tests/` | Small behavior checks with invented data. |

## Run the validation workflow

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then from this directory run `uv sync --locked`. The lockfile records the tested dependency versions. Substitute your own paths below; do not use official test responses for any of these commands.

1. Estimate training responses on raw training counts:

   ```bash
   uv run python scripts/prepare_training_effects.py --train /path/train.h5ad --output /path/new_training_output
   ```

2. Prepare human STRING v12.0 functional graph edges, channel support, and DSD neighbors. The validation file is used here for target labels only:

   ```bash
   uv run python scripts/prepare_networks.py --train-effects /path/new_training_output/train_effects.parquet --validation /path/validation.h5ad --string-info /path/9606.protein.info.v12.0.txt.gz --string-links /path/9606.protein.links.v12.0.txt.gz --string-detailed /path/9606.protein.links.detailed.v12.0.txt.gz --output /path/new_network_output
   ```

3. Score the common model family on validation:

   ```bash
   uv run python scripts/run_validation.py --train /path/train.h5ad --validation /path/validation.h5ad --train-effects /path/new_training_output/train_effects.parquet --edges /path/new_network_output/target_edges.csv --evidence-edges /path/new_network_output/evidence_edges.csv --dsd-neighbors /path/new_network_output/dsd_neighbors.csv --output /path/new_validation_output
   ```

   The default reports expression MAE and PDS on the selected number of cells per group, using the additive cell-generation assumption. `--generator hurdle` changes that assumption. `--de` adds differential-expression overlap; interpret it with the limitations in the assumptions document. Output directories must not already exist. Outputs include local cell barcodes, so do not commit them.

4. To use **every real validation cell** for expression MAE and PDS while staying within memory limits, run the same arguments with `scripts/run_all_validation_cells.py` instead. It streams all observed cells into group averages and uses 500 sampled training controls per predicted target by default. For these two metrics the group averages retain the needed information. It does not compute DE, which requires individual-cell variation.

Run small correctness checks with `uv run python -m unittest discover -s tests`. The graph, model and evaluation code have been compared with the local exploratory results; see the experiment notes. Verify source identity against the official VCC manifest before any final claims. The scripts rely on the operator to supply the appropriate split; they cannot identify every mislabeled test file.

Do not commit raw H5AD, STRING downloads, per-cell files, local caches, or generated outputs. No GitHub action is performed by these scripts.
