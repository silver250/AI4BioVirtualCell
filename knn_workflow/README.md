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
| `src/cellmates/full_de.py` | Memory-bounded CPU calculations for full-cell DE. |
| `scripts/` | Explicit training preprocessing, network preparation and validation runs. |
| `tests/` | Small behavior checks with invented data. |

## Suggested reading order

1. Start with `notebooks/model_codebook.py` for the question, models and results.
2. Read `preprocessing.py` to see how counts become one response vector per target.
3. Read `baselines.py`, `direct.py`, then `dsd.py` to see how each model makes a prediction.
4. Read `cells.py` and `evaluation.py` for the assumptions used to turn predictions into cells and compare them with validation data. `neighbor_cells.py` is a separate experimental alternative.
5. Read the scripts in order: `prepare_training_effects.py`, `prepare_networks.py`, then `run_all_validation_cells.py` and `run_full_de.py`. `run_validation.py` is the smaller sampled-cell option and supplies shared helper functions.

The comments in these files explain each major step. The `tests/` folder uses small invented examples to check behavior; the CSV in `examples/` is a saved result, not code to edit to change a model.

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

5. To calculate full-cell DE on CPU, use the same input arguments with `scripts/run_full_de.py`. It uses all real validation targets and controls, and the same 500 predicted cells per target. For a smaller check, add `--targets ACLY`; an independently calculated ACLY DE parquet can optionally be supplied with `--reference-de /path/reference.parquet`.

   The script needs temporary scratch disk roughly equal to `number of controls × number of genes × 4` bytes, plus at least 1.5 GB free reserve (about 2.8 GB plus reserve for our validation data). Temporary files are removed on normal exit and handled exceptions; if the process is killed, check for leftover `control_*` folders after confirming it has stopped. DE tables, sample identifiers, settings, and scores are local outputs and must not be committed. The calculation matches the documented `cell-eval2` 0.22.0 Scanpy/log-normalized settings, including its skipped CPM filter. See `docs/ASSUMPTIONS.md` before interpreting the scores.

Run small correctness checks with `uv run python -m unittest discover -s tests`. The graph, model and evaluation code have been compared with the local exploratory results; see the experiment notes. Verify source identity against the official VCC manifest before any final claims. The scripts rely on the operator to supply the appropriate split; they cannot identify every mislabeled test file.

Do not commit raw H5AD, STRING downloads, per-cell files, local caches, or generated outputs. No GitHub action is performed by these scripts.
