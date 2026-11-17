"""Compute full-validation DE overlap on CPU, one target and gene block at a time.

Run the training and network preparation scripts first. This script must receive
validation, not official test, responses. It writes per-target DE tables and
uses temporary disk space for the full control population; see the README.
"""

import argparse
import json
import os
import shutil
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import polars as pl
import psutil
from cell_eval2.metrics.de import de_overlap
from scipy.stats import false_discovery_control

from cellmates.baselines import predict_baselines
from cellmates.direct import predict_direct_knn
from cellmates.full_de import control_tie_sums, normalized_rows, wilcoxon_pvalues
from run_validation import dsd_effects, read_cells, sha256

GENE_BLOCK = 256
CELL_BLOCK = 512
CONTROL = "non-targeting"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "train",
        "validation",
        "train-effects",
        "edges",
        "evidence-edges",
        "dsd-neighbors",
        "output",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--cells", type=int, default=500, help="predicted cells per target"
    )
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument(
        "--targets", nargs="+", help="optional validation targets for a smaller check"
    )
    parser.add_argument(
        "--reference-de", help="optional independent ACLY DE parquet for a parity check"
    )
    return parser.parse_args()


def de_table(target, genes, p_values, fold_changes):
    """Adjust all gene p-values within the target and make a cell-eval2 table."""
    return pd.DataFrame(
        {
            "target": target,
            "feature": genes,
            "log2_fold_change": fold_changes,
            "p_value": p_values,
            "p_adj": false_discovery_control(p_values),
            "abs_log2_fold_change": np.abs(fold_changes),
        }
    )


def prepare_control_blocks(h5ad, codes, control_code, ranges, temp_dir):
    """Write normalized controls in gene blocks so the full matrix stays on disk."""
    handles = [open(temp_dir / f"{i:03d}.bin", "wb") for i in range(len(ranges))]
    written = 0
    try:
        for start in range(0, len(codes), CELL_BLOCK):
            stop = min(start + CELL_BLOCK, len(codes))
            mask = codes[start:stop] == control_code
            if not mask.any():
                continue
            control_cells = normalized_rows(h5ad, start, stop)[mask]
            for (left, right), handle in zip(ranges, handles):
                handle.write(control_cells[:, left:right].toarray().tobytes(order="C"))
            written += control_cells.shape[0]
    finally:
        for handle in handles:
            handle.close()
    return written


def sort_control_blocks(temp_dir, ranges, control_count, gene_count):
    """Sort controls by gene once and save values needed by each comparison."""
    tie_sums = np.empty(gene_count)
    control_means = np.empty(gene_count)
    for index, (left, right) in enumerate(ranges):
        path = temp_dir / f"{index:03d}.bin"
        controls = np.memmap(
            path, dtype=np.float32, mode="r", shape=(control_count, right - left)
        )
        sorted_controls = np.ascontiguousarray(np.sort(controls, axis=0).T)
        control_means[left:right] = np.expm1(sorted_controls).mean(
            axis=1, dtype=np.float64
        )
        tie_sums[left:right] = control_tie_sums(sorted_controls)
        sorted_path = temp_dir / f"{index:03d}.sorted"
        sorted_controls.tofile(sorted_path)
        del controls, sorted_controls
        os.replace(sorted_path, path)
    return tie_sums, control_means


def compare_reference(real_table, reference_path, genes):
    """Optionally verify ACLY against a separately calculated DE reference."""
    reference = pd.read_parquet(reference_path).set_index("feature").loc[genes]
    p_difference = float(
        np.max(np.abs(real_table.p_value.to_numpy() - reference.p_value.to_numpy()))
    )
    lfc_difference = float(
        np.max(
            np.abs(
                real_table.log2_fold_change.to_numpy()
                - reference.log2_fold_change.to_numpy()
            )
        )
    )
    differing_calls = int(
        np.sum(
            (real_table.p_adj.to_numpy() < 0.05) != (reference.p_adj.to_numpy() < 0.05)
        )
    )
    if p_difference > 1e-10 or lfc_difference > 1e-5 or differing_calls:
        raise ValueError("independent ACLY reference parity check failed")
    return {
        "max_p_diff": p_difference,
        "max_logfc_diff": lfc_difference,
        "different_significant": differing_calls,
    }


def main():
    args = parse_args()
    if version("cell-eval2") != "0.22.0":
        raise ValueError("full DE requires cell-eval2 0.22.0 for matching semantics")
    if args.cells < 1:
        raise ValueError("--cells must be positive")
    paths = {
        key: Path(getattr(args, key))
        for key in (
            "train",
            "validation",
            "train_effects",
            "edges",
            "evidence_edges",
            "dsd_neighbors",
        )
    }
    if paths["train"].resolve() == paths["validation"].resolve():
        raise ValueError("training and validation must be different files")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    tables = output / "tables"
    tables.mkdir()
    started = time.perf_counter()

    # 1. Build each predicted mean response from training targets only.
    training = pd.read_parquet(paths["train_effects"])
    controls, genes, sampled_ids = read_cells(
        paths["train"], [CONTROL], args.cells, np.random.default_rng(args.seed)
    )
    with h5py.File(paths["validation"], "r") as h5ad:
        labels = h5ad["obs/target_gene/categories"].asstr()[:]
        codes = h5ad["obs/target_gene/codes"][:]
        if (
            (codes < 0).any()
            or genes != h5ad["var/_index"].asstr()[:].tolist()
            or genes != training.columns.tolist()
        ):
            raise ValueError("target labels or gene order do not match")
        counts = pd.Series(np.bincount(codes, minlength=len(labels)), index=labels)
        all_targets = sorted(set(labels) - {CONTROL})
        if (
            CONTROL not in counts
            or len(all_targets) < 2
            or set(all_targets) & set(training.index)
        ):
            raise ValueError("invalid or overlapping training and validation targets")
        targets = sorted(args.targets) if args.targets else all_targets
        if len(set(targets)) != len(targets) or not set(targets) <= set(all_targets):
            raise ValueError("requested targets must be distinct validation targets")
        if args.reference_de and "ACLY" not in targets:
            raise ValueError("--reference-de requires ACLY among the selected targets")
        if args.reference_de:
            targets = ["ACLY"] + [target for target in targets if target != "ACLY"]

        edges = pd.read_csv(paths["edges"])
        evidence = pd.read_csv(paths["evidence_edges"])
        neighbors = pd.read_csv(paths["dsd_neighbors"])
        effects = predict_baselines(training, targets)
        for name, graph, weighting in (
            ("direct_equal", edges, "equal"),
            ("direct_score", edges, "score"),
            ("evidence_equal", evidence, "equal"),
        ):
            effects[name] = predict_direct_knn(
                training, targets, graph, k=5, min_score=0.4, weighting=weighting
            ).predictions.loc[targets]
        effects["dsd_equal"] = dsd_effects(training, targets, neighbors, 5)
        effects["dsd_equal_k10"] = dsd_effects(training, targets, neighbors, 10)

        # 2. Store normalized control cells on disk in small gene blocks.
        control_count = int(counts[CONTROL])
        control_code = int(np.flatnonzero(labels == CONTROL)[0])
        gene_count = len(genes)
        required_bytes = control_count * gene_count * 4
        if shutil.disk_usage(output).free < required_bytes + 1_500_000_000:
            raise RuntimeError("not enough scratch disk space plus 1.5 GB reserve")
        ranges = [
            (left, min(left + GENE_BLOCK, gene_count))
            for left in range(0, gene_count, GENE_BLOCK)
        ]
        metadata = {
            "status": "preparing_controls",
            "cell_eval2_version": version("cell-eval2"),
            "control_cells": control_count,
            "target_cells": counts.to_dict(),
            "targets": targets,
            "genes": gene_count,
            "predicted_cells_per_target": args.cells,
            "models": list(effects),
            "generator": "additive log-normalized shift, negative values clipped to zero",
            "sampling_seed": args.seed,
            "training_control_barcodes": sampled_ids,
            "method": "two-sided tie-corrected Wilcoxon; per-target Benjamini-Hochberg",
            "filter": "none: cell-eval2 0.22.0 Scanpy/lognorm skips the configured CPM filter",
            "sha256": {key: sha256(value) for key, value in paths.items()},
            "completed_targets": [],
        }
        if args.reference_de:
            metadata["reference_de_sha256"] = sha256(Path(args.reference_de))
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        with tempfile.TemporaryDirectory(prefix="control_", dir=output) as temporary:
            temp_dir = Path(temporary)
            written = prepare_control_blocks(
                h5ad, codes, control_code, ranges, temp_dir
            )
            if written != control_count:
                raise ValueError("control cell count does not match metadata")
            tie_sums, control_means = sort_control_blocks(
                temp_dir, ranges, control_count, gene_count
            )
            print(
                f"Prepared {control_count} controls for {len(targets)} targets",
                flush=True,
            )

            # 3. Compare each real and predicted target with all real controls.
            prediction_controls = controls[CONTROL]
            records = []
            for target in targets:
                if psutil.virtual_memory().available < 1_000_000_000:
                    raise MemoryError(
                        "less than 1 GB available; stopped before next target"
                    )
                target_start = time.perf_counter()
                real_cells, target_genes, _ = read_cells(
                    paths["validation"],
                    [target],
                    int(counts[target]),
                    np.random.default_rng(0),
                )
                if genes != target_genes:
                    raise ValueError("target gene order differs")
                target_cells = real_cells[target]
                predicted_effects = {
                    name: effect.loc[target].to_numpy(dtype=np.float32)
                    for name, effect in effects.items()
                }
                p_values = {
                    name: np.empty(gene_count) for name in ["real"] + list(effects)
                }
                fold_changes = {
                    name: np.empty(gene_count) for name in ["real"] + list(effects)
                }
                for index, (left, right) in enumerate(ranges):
                    control_block = np.memmap(
                        temp_dir / f"{index:03d}.bin",
                        dtype=np.float32,
                        mode="r",
                        shape=(right - left, control_count),
                    )
                    # The same controls are used for real cells and every model.
                    populations = {"real": target_cells[:, left:right]}
                    populations.update(
                        {
                            name: np.maximum(
                                prediction_controls[:, left:right] + effect[left:right],
                                0,
                            )
                            for name, effect in predicted_effects.items()
                        }
                    )
                    for name, cells in populations.items():
                        p_values[name][left:right] = wilcoxon_pvalues(
                            control_block, cells, tie_sums[left:right]
                        )
                        mean = np.expm1(cells).mean(axis=0, dtype=np.float64)
                        fold_changes[name][left:right] = np.log2(
                            (mean + 1e-9) / (control_means[left:right] + 1e-9)
                        )
                    del control_block

                real_table = de_table(
                    target, genes, p_values["real"], fold_changes["real"]
                )
                if target == "ACLY" and args.reference_de:
                    metadata["independent_full_control_parity"] = compare_reference(
                        real_table, args.reference_de, genes
                    )
                real_table.to_parquet(tables / f"{target}_real.parquet", index=False)
                for name in effects:
                    predicted_table = de_table(
                        target, genes, p_values[name], fold_changes[name]
                    )
                    predicted_table.to_parquet(
                        tables / f"{target}_{name}.parquet", index=False
                    )
                    score = de_overlap(
                        de_pred=pl.from_pandas(predicted_table),
                        de_real=pl.from_pandas(real_table),
                        k=None,
                    )
                    records.append(
                        {
                            "target": target,
                            "method": name,
                            "de_wilcoxon_overlap": score[target],
                            "real_significant_genes": int(
                                (real_table.p_adj < 0.05).sum()
                            ),
                            "predicted_significant_genes": int(
                                (predicted_table.p_adj < 0.05).sum()
                            ),
                        }
                    )
                pd.DataFrame(records).to_csv(output / "per_target.csv", index=False)
                metadata["status"] = "running"
                metadata["completed_targets"].append(target)
                metadata["elapsed_seconds"] = time.perf_counter() - started
                (output / "metadata.json").write_text(
                    json.dumps(metadata, indent=2) + "\n"
                )
                print(
                    f"Completed {target} ({len(metadata['completed_targets'])}/{len(targets)}) in {time.perf_counter() - target_start:.1f}s",
                    flush=True,
                )

            # 4. Summarize only after every selected target has completed.
            result = pd.DataFrame(records)
            result.groupby("method", as_index=False).de_wilcoxon_overlap.mean().to_csv(
                output / "summary.csv", index=False
            )
            metadata["status"] = "complete"
            metadata["elapsed_seconds"] = time.perf_counter() - started
            (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
