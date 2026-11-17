"""Run a reproducible validation comparison from explicit local data paths.

The input labels and observed cells must be the validation split, never test.
STRING edge and DSD neighbor files must be prepared without test responses.
"""

import argparse
import hashlib
import json
import time
import warnings
from collections import Counter
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from cellmates.baselines import predict_baselines
from cellmates.cells import generate_cells
from cellmates.direct import predict_direct_knn
from cellmates.evaluation import score_validation


def read_cells(path, targets, count, rng):
    """Sample each group and convert its raw counts to log-normalized cells.

    Sampling is without replacement. The returned cell IDs are kept locally in
    the run metadata so the exact sample can be checked later.
    """
    # Open read-only and select the same number of cells for every group.
    data = ad.read_h5ad(path, backed="r")
    try:
        if not data.var_names.is_unique or not data.obs_names.is_unique:
            raise ValueError("gene and cell identifiers must be unique")
        labels = data.obs["target_gene"].astype(str).to_numpy()
        selected = {}
        for target in targets:
            options = np.flatnonzero(labels == target)
            if len(options) < count:
                raise ValueError(f"{target} has fewer than {count} cells")
            selected[target] = np.sort(rng.choice(options, size=count, replace=False))
        # Load only the selected rows, then normalize each cell to 10,000.
        index = np.sort(np.concatenate(list(selected.values())))
        expression = data[index, :].X
        expression = (
            expression.toarray()
            if sparse.issparse(expression)
            else np.asarray(expression)
        )
        expression = expression.astype(float)
        if (
            not np.isfinite(expression).all()
            or (expression < 0).any()
            or not np.equal(expression, np.floor(expression)).all()
        ):
            raise ValueError("input X must contain finite nonnegative integer counts")
        totals = expression.sum(axis=1)
        if (totals <= 0).any():
            raise ValueError("zero-library cell")
        expression = np.log1p(expression / totals[:, None] * 10_000).astype(np.float32)
        return (
            {
                target: expression[np.searchsorted(index, idx)]
                for target, idx in selected.items()
            },
            list(data.var_names),
            {
                target: [str(cell_id) for cell_id in data.obs_names[idx]]
                for target, idx in selected.items()
            },
        )
    finally:
        data.file.close()


def sha256(path):
    """Identify the exact input file used for this run."""
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def dsd_effects(train, targets, neighbors, k):
    """Average the responses of the k closest DSD training neighbors."""
    if not {"target", "neighbor", "rank"} <= set(neighbors.columns):
        raise ValueError("DSD neighbors need target, neighbor, rank")
    if (
        neighbors.duplicated(["target", "neighbor"]).any()
        or neighbors.duplicated(["target", "rank"]).any()
    ):
        raise ValueError("duplicate DSD neighbors or ranks")
    if not set(neighbors.neighbor) <= set(train.index):
        raise ValueError("DSD neighbor missing from training")
    rows = []
    for target in targets:
        chosen = (
            neighbors[neighbors.target == target]
            .sort_values("rank")
            .head(k)
            .neighbor.tolist()
        )
        rows.append(train.loc[chosen].mean(axis=0) if chosen else train.mean(axis=0))
    return pd.DataFrame(rows, index=targets, columns=train.columns)


def run(args):
    # 1. Check file paths and ensure train and validation targets are disjoint.
    start = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        k: Path(getattr(args, k))
        for k in [
            "train",
            "validation",
            "train_effects",
            "edges",
            "evidence_edges",
            "dsd_neighbors",
        ]
    }
    if paths["train"].resolve() == paths["validation"].resolve():
        raise ValueError("train and validation paths must differ")
    train = pd.read_parquet(paths["train_effects"])
    valid = ad.read_h5ad(paths["validation"], backed="r")
    try:
        labels = set(valid.obs["target_gene"].astype(str))
        targets = sorted(labels - {"non-targeting"})
    finally:
        valid.file.close()
    if "non-targeting" not in labels or len(targets) < 2:
        raise ValueError("validation needs controls and at least two targets")
    if set(targets) & set(train.index):
        raise ValueError("train/validation target overlap")
    # 2. Sample the real cells and a common set of training control cells.
    rng = np.random.default_rng(args.seed)
    control, genes, ctrl_ids = read_cells(
        paths["train"], ["non-targeting"], args.cells, rng
    )
    real, real_genes, real_ids = read_cells(
        paths["validation"], ["non-targeting"] + targets, args.cells, rng
    )
    if genes != real_genes or genes != train.columns.tolist():
        raise ValueError("gene names and order must match")
    # 3. Make response vectors for the baselines and each network model.
    edge = pd.read_csv(paths["edges"])
    evidence = pd.read_csv(paths["evidence_edges"])
    dsd = pd.read_csv(paths["dsd_neighbors"])
    models = predict_baselines(train, targets)
    for name, graph, weight in [
        ("direct_equal", edge, "equal"),
        ("direct_score", edge, "score"),
        ("evidence_equal", evidence, "equal"),
    ]:
        result = predict_direct_knn(
            train, targets, graph, k=5, min_score=0.4, weighting=weight
        )
        models[name] = result.predictions.loc[targets]
        result.diagnostics.to_csv(output / f"{name}_coverage.csv")
        result.neighbors.to_csv(output / f"{name}_neighbors.csv", index=False)
    models["dsd_equal"] = dsd_effects(train, targets, dsd, 5)
    models["dsd_equal_k10"] = dsd_effects(train, targets, dsd, 10)
    # 4. Assemble the observed cells with target labels for cell-eval2.
    labels = ["non-targeting"] * args.cells + [
        target for target in targets for _ in range(args.cells)
    ]
    obs = pd.DataFrame(
        {"target_gene": labels}, index=[f"cell_{i}" for i in range(len(labels))]
    )
    var = pd.DataFrame(index=train.columns)
    observed = ad.AnnData(
        np.vstack([real["non-targeting"]] + [real[target] for target in targets]),
        obs=obs.copy(),
        var=var.copy(),
    )
    metadata = {
        "status": "running",
        "paths": {k: str(path.resolve()) for k, path in paths.items()},
        "sha256": {k: sha256(path) for k, path in paths.items()},
        "seed": args.seed,
        "generation_seed": args.generation_seed,
        "targets": targets,
        "cells_per_group": args.cells,
        "train_control_barcodes": ctrl_ids,
        "validation_barcodes": real_ids,
        "generator": args.generator,
        "include_de": args.de,
        "models": {},
    }
    # 5. Use the same control cells and random seed for every model. Convert
    # the predicted mean effects into cells, then score and save diagnostics.
    for name, effect in models.items():
        model_start = time.perf_counter()
        parts = [control["non-targeting"]]
        diags = []
        random = np.random.default_rng(args.generation_seed)
        for target in targets:
            activation_random = random.random(parts[0].shape)
            donor_random = random.random(parts[0].shape)
            cells, diagnostic = generate_cells(
                parts[0],
                effect.loc[target].to_numpy(),
                activation_random,
                donor_random,
                args.generator,
            )
            parts.append(cells)
            diags.append({"target": target, **diagnostic})
        predicted = ad.AnnData(np.vstack(parts), obs=obs.copy(), var=var.copy())
        # Record warnings instead of hiding them, especially when DE is used.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scores, aggregate, config = score_validation(
                predicted, observed, include_de=args.de
            )
        scores.write_csv(output / f"{name}_per_target.csv")
        aggregate.write_csv(output / f"{name}_aggregate.csv")
        pd.DataFrame(diags).to_csv(output / f"{name}_generation.csv", index=False)
        metadata["models"][name] = {
            "seconds": time.perf_counter() - model_start,
            "warnings": dict(
                Counter(f"{w.category.__name__}: {w.message}" for w in caught)
            ),
            "nonfinite_metrics": int((~np.isfinite(scores["value"].to_numpy())).sum()),
            "evaluation": config,
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str) + "\n"
        )
        print(name, flush=True)
    # Mark completion only after every model has been scored.
    metadata["status"] = "complete"
    metadata["seconds"] = time.perf_counter() - start
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in (
        "train",
        "validation",
        "train-effects",
        "edges",
        "evidence-edges",
        "dsd-neighbors",
        "output",
    ):
        parser.add_argument("--" + arg, required=True)
    parser.add_argument("--cells", type=int, default=50)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--generation-seed", type=int, default=90210)
    parser.add_argument(
        "--generator", choices=["additive", "hurdle"], default="additive"
    )
    parser.add_argument(
        "--de",
        action="store_true",
        help="Include DE; cell-distribution assumptions still under investigation",
    )
    args = parser.parse_args()
    if args.cells < 1:
        parser.error("--cells must be positive")
    run(args)


if __name__ == "__main__":
    main()
