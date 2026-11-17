"""Estimate one response vector for each training target.

Run this only on the training H5AD. Each response is the average expression
for a target minus the average expression for non-targeting controls.
"""

import argparse
import hashlib
import json
from pathlib import Path

from cellmates.preprocessing import target_means


def main():
    # The caller supplies a training file and a new, empty output location.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="training H5AD")
    parser.add_argument("--output", required=True, help="new output directory")
    args = parser.parse_args()
    training_file = Path(args.train)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    # Read cells in chunks, normalize them, and compute target-minus-control
    # means. Each row of effects is a target; each column is a measured gene.
    effects, cell_counts, total_cells = target_means(training_file)
    effects.to_parquet(output / "train_effects.parquet")
    cell_counts.rename_axis("target").to_csv(output / "train_cell_counts.csv")

    # Save the source fingerprint and transformation for reproducibility.
    with training_file.open("rb") as source:
        source_hash = hashlib.file_digest(source, "sha256").hexdigest()
    metadata = {
        "source": str(training_file.resolve()),
        "sha256": source_hash,
        "cells": total_cells,
        "normalization": (
            "per-cell count total 10000, log1p, then target mean minus control mean"
        ),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
