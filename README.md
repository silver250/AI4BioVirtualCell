# AI4Bio Virtual Cell

We are predicting gene expression responses to unseen single-gene CRISPRi knockdowns in the public VCC 2025 H1 stem cell dataset. The main comparison is no change, mean effect, STRING nearest neighbors, and a small neural network. See [the analysis plan](docs/PROJECT_PLAN.md) for precise methods and split rules.

## Where to work

Work from this repository root, the folder containing this README. The shared marimo notebook is [`ai4bio-virtualcell-project.py`](ai4bio-virtualcell-project.py). Use it for interactive exploration; move reusable preprocessing, models, and scoring code into `src/` as those functions are developed. Put separate experiments in separate notebooks or scripts. Avoid simultaneous edits to the same notebook because cell changes can be difficult to merge.

| Location | Purpose | Commit? |
| --- | --- | --- |
| `docs/` | Research decisions and handoff notes | Yes |
| `data/raw/` | Downloaded VCC and STRING data | No |
| `data/processed/` | Derived matrices and caches | No |
| `results/` | Small score tables and figures | Yes |
| `results/predictions/` | Large prediction files | No |

The dataset is not in this repository. Use the [Arc VCC 2025 data page](https://github.com/ArcInstitute/arc-virtual-cell-atlas/tree/main/virtual-cell-challenge). VCC 2026 control files do not contain the 2025 perturbation responses needed for this study.

## Start the notebook

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
uv run marimo edit ai4bio-virtualcell-project.py
```

Install analysis packages with `uv sync --extra analysis`; add PyTorch for the neural network with `uv sync --extra analysis --extra dl`. When dependency resolution is available, generate and commit `uv.lock` so collaborators use identical versions. [Marimo's project guide](https://docs.marimo.io/guides/package_management/projects/) describes this shared-environment workflow.

## Collaboration routine

1. Pull `main` and create one short branch per task, such as `data-audit`, `string-knn`, `mlp`, or `evaluation`.
2. Agree on the target splits and prediction-file format before model work. Record seeds and settings in versioned files as they are created.
3. Commit code, documentation, and small results. Keep raw data, caches, credentials, and large predictions out of Git.
4. Open a pull request for review. Include what ran, the target set, and a score table or the current error.

Suggested work areas are data and split audit, STRING similarity, the neural network, and scoring. Track each person's current owner and status in issues or pull requests.
