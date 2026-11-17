"""Prepare train/validation STRING edges and DSD neighbors without test responses."""

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from cellmates.dsd import predict_dsd


def file_sha256(path):
    """Fingerprint an input file so a run can be reproduced."""
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    # 1. Read paths for the training effects, validation labels, and STRING.
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "train-effects",
        "validation",
        "string-info",
        "string-links",
        "string-detailed",
        "output",
    ):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=False)
    # 2. Read only target labels from validation; never use its responses here.
    train = pd.read_parquet(args.train_effects)
    validation = ad.read_h5ad(args.validation, backed="r")
    try:
        targets = sorted(
            set(validation.obs["target_gene"].astype(str)) - {"non-targeting"}
        )
    finally:
        validation.file.close()
    if not targets or set(targets) & set(train.index):
        raise ValueError("invalid target labels or overlap")
    all_targets = set(targets) | set(train.index)
    # 3. Map every target to exactly one STRING preferred name.
    # Ambiguous or missing matches stop the run instead of guessing an alias.
    matches_by_name = defaultdict(list)
    protein_ids = []
    with gzip.open(args.string_info, "rt") as f:
        for record in csv.DictReader(f, delimiter="\t"):
            protein_ids.append(record["#string_protein_id"])
            if record["preferred_name"] in all_targets:
                matches_by_name[record["preferred_name"]].append(
                    record["#string_protein_id"]
                )
    if len(protein_ids) != len(set(protein_ids)) or any(
        len(matches_by_name[t]) != 1 for t in all_targets
    ):
        raise ValueError(
            "STRING IDs must be unique; every target needs one exact preferred-name match"
        )
    protein_ids.sort()
    protein_index = {v: i for i, v in enumerate(protein_ids)}
    gene_to_id = {t: matches_by_name[t][0] for t in all_targets}
    id_to_gene = {v: k for k, v in gene_to_id.items()}
    if len(id_to_gene) != len(gene_to_id):
        raise ValueError("multiple targets map to one protein")
    pd.DataFrame(
        [
            {
                "target": t,
                "string_id": gene_to_id[t],
                "split": "train" if t in train.index else "validation",
            }
            for t in sorted(all_targets)
        ]
    ).to_csv(output_dir / "target_mapping.csv", index=False)
    # 4. Build the full undirected protein graph. STRING stores scores from
    # 0 to 1000, so 400 is the 0.4 cutoff and division by 1000 gives 0–1.
    rows = []
    cols = []
    weights = []
    for chunk in pd.read_csv(args.string_links, sep=r"\s+", chunksize=500_000):
        keep = chunk[chunk.combined_score >= 400]
        left = keep.protein1.map(protein_index)
        right = keep.protein2.map(protein_index)
        if left.isna().any() or right.isna().any():
            raise ValueError("network protein absent from info")
        rows.append(left.to_numpy(dtype=np.int32))
        cols.append(right.to_numpy(dtype=np.int32))
        weights.append(keep.combined_score.to_numpy(dtype=float) / 1000)
    frame = pd.DataFrame(
        {
            "a": np.concatenate(rows),
            "b": np.concatenate(cols),
            "score": np.concatenate(weights),
        }
    )
    # Keep the largest score for duplicate or reversed records.
    frame = frame.groupby(["a", "b"], as_index=False, sort=False).score.max()
    graph = sparse.csr_matrix(
        (frame.score.to_numpy(), (frame["a"].to_numpy(), frame["b"].to_numpy())),
        shape=(len(protein_ids), len(protein_ids)),
    )
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    node_by_gene = {t: protein_index[s] for t, s in gene_to_id.items()}
    # Pull only target-to-target edges from the graph; rows are undirected once.
    target_names = sorted(all_targets)
    target_nodes = [node_by_gene[t] for t in target_names]
    sub = sparse.triu(graph[target_nodes, :][:, target_nodes], k=1).tocoo()
    edges = pd.DataFrame(
        {
            "gene_a": [target_names[i] for i in sub.row],
            "gene_b": [target_names[i] for i in sub.col],
            "score": sub.data,
        }
    )
    edges.to_csv(output_dir / "target_edges.csv", index=False)
    # 5. Apply the evidence support rule to target-to-target edges. Other
    # evidence still contributes to the combined score used for ranking.
    detailed = {}
    with gzip.open(args.string_detailed, "rt") as f:
        header = next(f).split()
        ix = {
            n: header.index(n)
            for n in [
                "protein1",
                "protein2",
                "experimental",
                "database",
                "combined_score",
            ]
        }
        for line in f:
            fields = line.split()
            protein_a, protein_b = fields[ix["protein1"]], fields[ix["protein2"]]
            if (
                protein_a not in id_to_gene
                or protein_b not in id_to_gene
                or protein_a == protein_b
            ):
                continue
            experimental_score, database_score, score = (
                int(fields[ix[n]])
                for n in ("experimental", "database", "combined_score")
            )
            if score < 400 or not (experimental_score > 0 or database_score > 0):
                continue
            pair = tuple(sorted((id_to_gene[protein_a], id_to_gene[protein_b])))
            old = detailed.get(pair)
            if old is not None and old != (experimental_score, database_score, score):
                raise ValueError("inconsistent directed evidence duplicates")
            detailed[pair] = (experimental_score, database_score, score)
    evidence_edges = pd.DataFrame(
        [
            {
                "gene_a": protein_a,
                "gene_b": protein_b,
                "experimental": e / 1000,
                "database": d / 1000,
                "score": s / 1000,
            }
            for (protein_a, protein_b), (e, d, s) in sorted(detailed.items())
        ],
        columns=["gene_a", "gene_b", "experimental", "database", "score"],
    )
    evidence_edges.to_csv(output_dir / "evidence_edges.csv", index=False)
    # 6. Find DSD neighbors using all proteins as possible intermediates.
    # One dummy response is sufficient: neighbor identities depend only on the graph.
    dummy = pd.DataFrame({"placeholder": np.zeros(len(train))}, index=train.index)
    _, diagnostics, neighbors = predict_dsd(
        dummy, targets, node_by_gene, graph, k=10, steps=6
    )
    neighbors.to_csv(output_dir / "dsd_neighbors.csv", index=False)
    diagnostics.to_csv(output_dir / "dsd_coverage.csv")
    # 7. Save source checksums and graph settings for the audit trail.
    paths = {
        "train_effects": args.train_effects,
        "validation": args.validation,
        "string_info": args.string_info,
        "string_links": args.string_links,
        "string_detailed": args.string_detailed,
    }
    metadata = {
        "sha256": {k: file_sha256(Path(v)) for k, v in paths.items()},
        "release": "STRING human v12.0 functional combined",
        "cutoff": 0.4,
        "dsd_steps": 6,
        "k": 10,
        "nodes": len(protein_ids),
        "edges": graph.nnz // 2,
        "evidence_filter": "combined >= 0.4 AND (experimental > 0 OR database > 0)",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
