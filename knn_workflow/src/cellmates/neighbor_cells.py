"""Experimental transfer of observed training-neighbor cells.

This is a different model from adding a predicted mean effect to controls.
"""
import numpy as np


def mix_neighbor_cells(pools, neighbors, weights, n_cells, rng):
    """Sample observed source cells without replacement within each source.

    ``pools`` maps target names to 2-D arrays on the common expression scale.
    Reuse of source cells across different predicted targets remains possible.
    """
    names=list(neighbors)
    w=np.asarray(weights,dtype=float)
    if not names or len(set(names))!=len(names) or w.shape!=(len(names),):
        raise ValueError('neighbors must be unique and match weights')
    if n_cells<1 or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:
        raise ValueError('invalid cell count or weights')
    exact=w/w.sum()*n_cells
    counts=np.floor(exact).astype(int)
    order=np.argsort(-(exact-counts),kind='stable')
    counts[order[:n_cells-counts.sum()]]+=1
    pieces=[]
    for name,n in zip(names,counts):
        cells=np.asarray(pools[name])
        if cells.ndim!=2 or n>len(cells):
            raise ValueError(f'not enough source cells for {name}')
        if n:
            pieces.append(cells[rng.choice(len(cells),size=n,replace=False)])
    return np.vstack(pieces),dict(zip(names,counts.tolist()))
