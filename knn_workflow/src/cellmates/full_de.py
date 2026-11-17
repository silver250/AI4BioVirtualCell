"""Memory-bounded Wilcoxon calculations for the full-cell DE comparison.

The implementation matches the Scanpy/log-normalized settings used with
cell-eval2 0.22.0. It works on blocks of genes so the entire validation matrix
does not need to fit in memory. See docs/ASSUMPTIONS.md for the limitations.
"""

import numpy as np
from numba import njit, prange
from scipy import sparse
from scipy.special import ndtr


def normalized_rows(h5ad, start, stop):
    """Read raw CSR rows and return log1p library-normalized sparse cells."""
    pointers = h5ad["X/indptr"][start : stop + 1].astype(np.int64)
    values = h5ad["X/data"][pointers[0] : pointers[-1]].astype(np.float64)
    indices = h5ad["X/indices"][pointers[0] : pointers[-1]]
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or not np.equal(values, np.floor(values)).all()
    ):
        raise ValueError("X must contain finite nonnegative integer counts")

    # Normalize each cell before comparing target and control groups.
    cells = sparse.csr_matrix(
        (values, indices, pointers - pointers[0]),
        shape=(stop - start, int(h5ad["X"].attrs["shape"][1])),
    )
    totals = np.asarray(cells.sum(axis=1)).ravel()
    if (totals <= 0).any():
        raise ValueError("found a cell with zero total counts")
    cells.data *= np.repeat(10_000 / totals, np.diff(cells.indptr))
    np.log1p(cells.data, out=cells.data)
    return cells.astype(np.float32)


@njit(parallel=True)
def control_tie_sums(sorted_controls):
    """Calculate the repeated-value correction for each control gene."""
    gene_count, control_count = sorted_controls.shape
    tie_sums = np.empty(gene_count, dtype=np.float64)
    for gene in prange(gene_count):
        values = sorted_controls[gene]
        total = 0.0
        start = 0
        while start < control_count:
            stop = start + 1
            while stop < control_count and values[stop] == values[start]:
                stop += 1
            size = float(stop - start)
            total += size * size * size - size
            start = stop
        tie_sums[gene] = total
    return tie_sums


@njit(parallel=True)
def _rank_statistics(sorted_controls, target_cells, control_ties):
    """Calculate a tie-corrected rank statistic for every gene in a block."""
    target_count = target_cells.shape[0]
    control_count = sorted_controls.shape[1]
    gene_count = target_cells.shape[1]
    statistics = np.empty(gene_count, dtype=np.float64)

    for gene in prange(gene_count):
        controls = sorted_controls[gene]
        targets = np.sort(target_cells[:, gene])
        tie_sum = control_ties[gene]
        rank_sum = 0.0
        start = 0
        while start < target_count:
            stop = start + 1
            while stop < target_count and targets[stop] == targets[start]:
                stop += 1
            value = targets[start]
            target_ties = float(stop - start)
            lower = np.searchsorted(controls, value, "left")
            upper = np.searchsorted(controls, value, "right")
            control_ties_for_value = float(upper - lower)
            rank_sum += target_ties * (lower + 0.5 * control_ties_for_value)
            combined = control_ties_for_value + target_ties
            tie_sum += (combined**3 - combined) - (
                control_ties_for_value**3 - control_ties_for_value
            )
            start = stop

        total = float(target_count + control_count)
        tie_correction = 1.0 - tie_sum / (total**3 - total)
        standard_deviation = np.sqrt(
            tie_correction * target_count * control_count * (total + 1.0) / 12.0
        )
        # Fully tied genes have no meaningful difference; Scanpy assigns z=0.
        statistics[gene] = (
            0.0
            if standard_deviation == 0.0
            else (rank_sum - target_count * control_count / 2.0) / standard_deviation
        )
    return statistics


def wilcoxon_pvalues(sorted_controls, target_cells, control_ties=None):
    """Return two-sided Wilcoxon p-values; controls are genes by sorted cells."""
    if control_ties is None:
        control_ties = control_tie_sums(sorted_controls)
    statistics = _rank_statistics(sorted_controls, target_cells, control_ties)
    return 2 * ndtr(-np.abs(statistics))
