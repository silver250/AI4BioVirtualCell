"""Experimental transfer of observed training-neighbor cells.

This is a different model from adding a predicted mean effect to controls.
"""

import numpy as np


def mix_neighbor_cells(pools, neighbors, weights, n_cells, rng):
    """Sample observed source cells without replacement within each source.

    ``pools`` maps target names to 2-D arrays on the common expression scale.
    Reuse of source cells across different predicted targets remains possible.
    """
    # 1. Check that each neighbor has one nonnegative weight.
    names = list(neighbors)
    neighbor_weights = np.asarray(weights, dtype=float)
    if (
        not names
        or len(set(names)) != len(names)
        or neighbor_weights.shape != (len(names),)
    ):
        raise ValueError("neighbors must be unique and match weights")
    if (
        n_cells < 1
        or not np.isfinite(neighbor_weights).all()
        or (neighbor_weights < 0).any()
        or neighbor_weights.sum() <= 0
    ):
        raise ValueError("invalid cell count or weights")
    # 2. Divide the requested cell count among neighbors using their weights.
    # Largest remainders make the integer counts add up to n_cells.
    exact = neighbor_weights / neighbor_weights.sum() * n_cells
    counts = np.floor(exact).astype(int)
    order = np.argsort(-(exact - counts), kind="stable")
    counts[order[: n_cells - counts.sum()]] += 1
    # 3. Sample observed training cells without replacement within each pool.
    pieces = []
    for name, number_to_sample in zip(names, counts):
        cells = np.asarray(pools[name])
        if cells.ndim != 2 or number_to_sample > len(cells):
            raise ValueError(f"not enough source cells for {name}")
        if number_to_sample:
            pieces.append(
                cells[rng.choice(len(cells), size=number_to_sample, replace=False)]
            )
    return np.vstack(pieces), dict(zip(names, counts.tolist()))
