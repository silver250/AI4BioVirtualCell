"""Estimate target response vectors from raw training counts."""

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


def target_means(path, chunk_size=512, library_size=10000, include_means=False):
    """Stream raw counts and return each target's mean response to control.

    Normalize each cell to library_size, apply log1p, then average by target.
    When include_means is True, also return the actual group expression means.
    The input must have sparse CSR X and the documented target_gene metadata.
    """
    with h5py.File(path, "r") as h5ad:
        shape = tuple(h5ad["X"].attrs["shape"])
        genes = h5ad["var/_index"].asstr()[:]
        labels = h5ad["obs/target_gene/categories"].asstr()[:]
        codes = h5ad["obs/target_gene/codes"][:].astype(int)
        if len(genes) != len(set(genes)) or (codes < 0).any():
            raise ValueError("Expected unique gene names and nonmissing target labels")
        # Accumulate a sum for each target without loading all cells at once.
        sums = np.zeros((len(labels), shape[1]), dtype=np.float64)
        counts = np.bincount(codes, minlength=len(labels))
        zero_cells = 0
        for start in range(0, shape[0], chunk_size):
            stop = min(start + chunk_size, shape[0])
            # Read one contiguous set of sparse rows from the H5AD file.
            pointers = h5ad["X/indptr"][start : stop + 1].astype(np.int64)
            values = h5ad["X/data"][pointers[0] : pointers[-1]].astype(np.float64)
            indices = h5ad["X/indices"][pointers[0] : pointers[-1]]
            if (
                not np.isfinite(values).all()
                or (values < 0).any()
                or not np.equal(values, np.floor(values)).all()
            ):
                raise ValueError("Expected finite nonnegative integer raw counts in X")
            cell_counts = sparse.csr_matrix(
                (values, indices, pointers - pointers[0]),
                shape=(stop - start, shape[1]),
            )
            # Normalize each cell before averaging; zero-library cells fail.
            totals = np.asarray(cell_counts.sum(axis=1)).ravel()
            zero_cells += int((totals == 0).sum())
            scale = np.divide(
                library_size, totals, out=np.zeros_like(totals), where=totals > 0
            )
            cell_counts.data *= np.repeat(scale, np.diff(cell_counts.indptr))
            np.log1p(cell_counts.data, out=cell_counts.data)
            # Add each cell to the sum of its assigned target.
            selector = sparse.csr_matrix(
                (np.ones(stop - start), (codes[start:stop], np.arange(stop - start))),
                shape=(len(labels), stop - start),
            )
            sums += (selector @ cell_counts).toarray()
            if start % (512 * 50) == 0:
                print(f"{path.name}: {stop}/{shape[0]} cells", flush=True)
        if zero_cells or (counts == 0).any():
            raise ValueError(
                f"Found {zero_cells} zero-library cells or empty target categories"
            )
        # Divide by cell counts, then subtract the control mean gene by gene.
        means = pd.DataFrame(sums / counts[:, None], index=labels, columns=genes)
        if "non-targeting" not in means.index:
            raise ValueError("Missing non-targeting control")
        effects = means.subtract(means.loc["non-targeting"], axis="columns").drop(
            index="non-targeting"
        )
        result = (effects, pd.Series(counts, index=labels, name="cells"), int(shape[0]))
        return (*result, means) if include_means else result
