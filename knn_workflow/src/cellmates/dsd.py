"""Finite-step diffusion state distance on a weighted, undirected graph."""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def transition_matrix(adjacency):
    a = sparse.csr_matrix(adjacency, dtype=np.float64)
    if a.shape[0] != a.shape[1] or not np.isfinite(a.data).all() or (a.data < 0).any():
        raise ValueError('adjacency must be square, finite, nonnegative')
    if (a - a.T).nnz and np.max(np.abs((a - a.T).data)) > 1e-12:
        raise ValueError('adjacency must be symmetric')
    a = a.copy()
    a.setdiag(0)
    a.eliminate_zeros()
    degree = np.asarray(a.sum(axis=1)).ravel()
    isolated = np.flatnonzero(degree == 0)
    if len(isolated):
        a = a + sparse.csr_matrix((np.ones(len(isolated)), (isolated, isolated)), shape=a.shape)
    degree[isolated] = 1
    return (sparse.diags(1 / degree) @ a).tocsr()


def visitation_profiles(transition, starts, steps=6):
    """Rows are sum of state distributions at times 0 through steps."""
    p = sparse.csr_matrix(transition, dtype=np.float64)
    if p.shape[0] != p.shape[1] or not np.isfinite(p.data).all() or (p.data < 0).any():
        raise ValueError('invalid transition matrix')
    if not np.allclose(np.asarray(p.sum(axis=1)).ravel(), 1, atol=1e-10):
        raise ValueError('transition rows must sum to one')
    if not isinstance(steps, int) or steps < 0:
        raise ValueError('steps must be a nonnegative integer')
    starts = np.asarray(starts, dtype=int)
    if starts.ndim != 1 or (starts < 0).any() or (starts >= p.shape[0]).any():
        raise ValueError('invalid start indices')
    state = np.zeros((p.shape[0], len(starts)), dtype=np.float64)
    state[starts, np.arange(len(starts))] = 1
    visits = state.copy()
    for _ in range(steps):
        state = p.T @ state
        visits += state
    return visits.T


def predict_dsd(train_effects, query_targets, target_nodes, adjacency, k=5, steps=6, tol=1e-10):
    if not isinstance(k, int) or k < 1:
        raise ValueError('k must be positive')
    query = list(query_targets)
    if not query or len(query) != len(set(query)) or not train_effects.index.is_unique:
        raise ValueError('targets must be unique and nonempty')
    if set(query) & set(train_effects.index):
        raise ValueError('train and query targets must be disjoint')
    if train_effects.empty or not train_effects.columns.is_unique or not np.isfinite(train_effects.to_numpy()).all():
        raise ValueError('invalid training effects')
    names = list(train_effects.index) + query
    if any(x not in target_nodes for x in names) or len({target_nodes[x] for x in names}) != len(names):
        raise ValueError('missing or ambiguous target mapping')
    p = transition_matrix(adjacency)
    _, components = connected_components(p, directed=False)
    nodes = [target_nodes[x] for x in names]
    profiles = visitation_profiles(p, nodes, steps)
    n_train = len(train_effects)
    predictions, diagnostics, neighbors = [], [], []
    fallback = train_effects.mean(axis=0)
    for qi, target in enumerate(query):
        d = np.abs(profiles[:n_train] - profiles[n_train + qi]).sum(axis=1)
        valid = (components[nodes[:n_train]] == components[nodes[n_train + qi]]) & (d < 2 * (steps + 1) - tol)
        candidates = sorted(((float(d[i]), names[i]) for i in np.flatnonzero(valid)), key=lambda x: (x[0], x[1]))
        selected = candidates[:k]
        predictions.append(train_effects.loc[[name for _, name in selected]].mean(axis=0) if selected else fallback)
        diagnostics.append({'target': target, 'eligible_neighbors': len(candidates), 'neighbors_used': len(selected),
                            'fallback': not bool(selected), 'component': int(components[nodes[n_train + qi]]),
                            'weighted_degree': float(adjacency[nodes[n_train + qi]].sum())})
        for rank, (distance, name) in enumerate(selected, 1):
            neighbors.append({'target': target, 'rank': rank, 'neighbor': name, 'dsd': distance,
                              'weight': 1 / len(selected)})
    return (pd.DataFrame(predictions, index=pd.Index(query, name='target')),
            pd.DataFrame(diagnostics).set_index('target'),
            pd.DataFrame(neighbors, columns=['target', 'rank', 'neighbor', 'dsd', 'weight']))
