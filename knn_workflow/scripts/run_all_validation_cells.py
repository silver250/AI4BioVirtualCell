"""Score expression/PDS on all validation cells through streamed group means.

DE cannot be reconstructed from group means and is deliberately excluded.
"""

import argparse
import json
import time
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from cellmates.baselines import predict_baselines
from cellmates.direct import predict_direct_knn
from cellmates.evaluation import score_validation
from cellmates.preprocessing import target_means
from run_validation import read_cells, dsd_effects, sha256


def main():
    # 1. Read the explicit data paths and sampling settings.
    parser = argparse.ArgumentParser(description=__doc__)
    for n in (
        "train",
        "validation",
        "train-effects",
        "edges",
        "evidence-edges",
        "dsd-neighbors",
        "output",
    ):
        parser.add_argument("--" + n, required=True)
    parser.add_argument("--predicted-cells", type=int, default=500)
    parser.add_argument("--seed", type=int, default=420)
    args = parser.parse_args()
    if args.predicted_cells < 1:
        parser.error("--predicted-cells must be positive")
    started = time.perf_counter()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        k: Path(getattr(args, k))
        for k in (
            "train",
            "validation",
            "train_effects",
            "edges",
            "evidence_edges",
            "dsd_neighbors",
        )
    }
    if paths["train"].resolve() == paths["validation"].resolve():
        raise ValueError("train and validation paths must differ")
    # 2. Stream every validation cell into one mean expression per target.
    # MAE and PDS use these group means; DE cannot be recovered from them.
    train = pd.read_parquet(paths["train_effects"])
    _, counts, n_cells, means = target_means(paths["validation"], include_means=True)
    targets = sorted(set(means.index) - {"non-targeting"})
    if len(targets) < 2 or set(targets) & set(train.index):
        raise ValueError("invalid or overlapping targets")
    # 3. Sample the same training control cells for every model and target.
    controls, genes, ids = read_cells(
        paths["train"],
        ["non-targeting"],
        args.predicted_cells,
        np.random.default_rng(args.seed),
    )
    if genes != means.columns.tolist() or genes != train.columns.tolist():
        raise ValueError("gene names and order must match")
    # 4. Predict a response vector using each baseline and network model.
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
        result.diagnostics.to_csv(output_dir / f"{name}_coverage.csv")
        result.neighbors.to_csv(output_dir / f"{name}_neighbors.csv", index=False)
    models["dsd_equal"] = dsd_effects(train, targets, dsd, 5)
    models["dsd_equal_k10"] = dsd_effects(train, targets, dsd, 10)
    # 5. Represent each observed group by its already-computed mean.
    labels = ["non-targeting"] + targets
    obs = pd.DataFrame(
        {"target_gene": labels}, index=[f"group_{i}" for i in range(len(labels))]
    )
    var = pd.DataFrame(index=train.columns)
    observed = ad.AnnData(
        means.loc[labels].to_numpy(dtype=np.float64), obs=obs.copy(), var=var.copy()
    )
    metadata = {
        "status": "running",
        "analysis": "all real validation cells reduced to per-target mean log-normalized expression",
        "note": "Group means are sufficient for expr_mae and pds_l1 in this configuration; DE cannot be computed from group means.",
        "validation_cells": n_cells,
        "validation_cells_per_target": counts.to_dict(),
        "n_measured_genes": len(genes),
        "predicted_cells_per_target": args.predicted_cells,
        "prediction_population": "sampled training controls shifted by predicted log-scale effect and clipped at zero",
        "seed": args.seed,
        "training_control_barcodes": ids,
        "sha256": {k: sha256(v) for k, v in paths.items()},
        "models": {},
    }
    # 6. Shift the sampled controls, clip negative values, and average the
    # predicted cells. Score each model against the full validation means.
    control_cells = controls["non-targeting"]
    for name, effects in models.items():
        model_start = time.perf_counter()
        predicted_means = [control_cells.mean(axis=0, dtype=np.float64)]
        clipped = 0
        for target in targets:
            shifted_cells = control_cells + effects.loc[target].to_numpy(
                dtype=np.float32
            )
            clipped += int((shifted_cells < 0).sum())
            predicted_means.append(
                np.maximum(shifted_cells, 0).mean(axis=0, dtype=np.float64)
            )
        predicted = ad.AnnData(
            np.vstack(predicted_means), obs=obs.copy(), var=var.copy()
        )
        scores, aggregate, config = score_validation(
            predicted, observed, include_de=False
        )
        scores.write_csv(output_dir / f"{name}_per_target.csv")
        aggregate.write_csv(output_dir / f"{name}_aggregate.csv")
        metadata["models"][name] = {
            "seconds": time.perf_counter() - model_start,
            "evaluation": config,
            "fraction_clipped": clipped
            / (len(targets) * args.predicted_cells * len(genes)),
            "nonfinite_metrics": int((~np.isfinite(scores["value"].to_numpy())).sum()),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str) + "\n"
        )
        print("Scored", name, flush=True)
    # Record inputs, settings, scores and completion for later review.
    metadata["status"] = "complete"
    metadata["seconds"] = time.perf_counter() - started
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )


if __name__ == "__main__":
    main()
