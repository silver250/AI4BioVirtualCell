"""Finite-step diffusion state distance on a weighted, undirected graph."""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def transition_matrix(adjacency):
    """Turn undirected edge weights into probabilities for one walk step.

    A walk follows each outgoing edge in proportion to its STRING score.
    An isolated protein gets a self-loop so its transition row sums to one.
    """
    graph = sparse.csr_matrix(adjacency, dtype=np.float64)
    if (
        graph.shape[0] != graph.shape[1]
        or not np.isfinite(graph.data).all()
        or (graph.data < 0).any()
    ):
        raise ValueError("adjacency must be square, finite, nonnegative")
    if (graph - graph.T).nnz and np.max(np.abs((graph - graph.T).data)) > 1e-12:
        raise ValueError("adjacency must be symmetric")
    # Remove self-edges, then give isolated proteins a valid transition.
    graph = graph.copy()
    graph.setdiag(0)
    graph.eliminate_zeros()
    degree = np.asarray(graph.sum(axis=1)).ravel()
    isolated = np.flatnonzero(degree == 0)
    if len(isolated):
        graph = graph + sparse.csr_matrix(
            (np.ones(len(isolated)), (isolated, isolated)), shape=graph.shape
        )
    degree[isolated] = 1
    return (sparse.diags(1 / degree) @ graph).tocsr()


def visitation_profiles(transition, starts, steps=6):
    """Rows are sum of state distributions at times 0 through steps."""
    transition_matrix_for_walk = sparse.csr_matrix(transition, dtype=np.float64)
    if (
        transition_matrix_for_walk.shape[0] != transition_matrix_for_walk.shape[1]
        or not np.isfinite(transition_matrix_for_walk.data).all()
        or (transition_matrix_for_walk.data < 0).any()
    ):
        raise ValueError("invalid transition matrix")
    if not np.allclose(
        np.asarray(transition_matrix_for_walk.sum(axis=1)).ravel(), 1, atol=1e-10
    ):
        raise ValueError("transition rows must sum to one")
    if not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    starts = np.asarray(starts, dtype=int)
    if (
        starts.ndim != 1
        or (starts < 0).any()
        or (starts >= transition_matrix_for_walk.shape[0]).any()
    ):
        raise ValueError("invalid start indices")
    # Each column tracks a walk starting at one training or query target.
    state = np.zeros(
        (transition_matrix_for_walk.shape[0], len(starts)), dtype=np.float64
    )
    state[starts, np.arange(len(starts))] = 1
    # Add expected visits at the starting point and after every step.
    visits = state.copy()
    for _ in range(steps):
        state = transition_matrix_for_walk.T @ state
        visits += state
    return visits.T


def predict_dsd(
    train_effects, query_targets, target_nodes, adjacency, k=5, steps=6, tol=1e-10
):
    """Predict using training targets with the closest walk-visit patterns.

    Distance is the sum of absolute differences between two visit profiles.
    Only overlapping profiles in the same component can be neighbors. Average
    up to k response vectors equally; use the training mean if none qualify.
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be positive")
    query = list(query_targets)
    if not query or len(query) != len(set(query)) or not train_effects.index.is_unique:
        raise ValueError("targets must be unique and nonempty")
    if set(query) & set(train_effects.index):
        raise ValueError("train and query targets must be disjoint")
    if (
        train_effects.empty
        or not train_effects.columns.is_unique
        or not np.isfinite(train_effects.to_numpy()).all()
    ):
        raise ValueError("invalid training effects")
    names = list(train_effects.index) + query
    if any(x not in target_nodes for x in names) or len(
        {target_nodes[x] for x in names}
    ) != len(names):
        raise ValueError("missing or ambiguous target mapping")
    # 1. Build the walk and compute profiles using the full protein graph.
    transition_matrix_for_walk = transition_matrix(adjacency)
    _, components = connected_components(transition_matrix_for_walk, directed=False)
    nodes = [target_nodes[x] for x in names]
    profiles = visitation_profiles(transition_matrix_for_walk, nodes, steps)
    num_training_targets = len(train_effects)
    predictions, diagnostics, neighbors = [], [], []
    fallback = train_effects.mean(axis=0)
    # 2. Rank only training targets, then average their observed responses.
    for query_index, target in enumerate(query):
        distances = np.abs(
            profiles[:num_training_targets]
            - profiles[num_training_targets + query_index]
        ).sum(axis=1)
        # Disjoint visit profiles have the maximum possible distance.
        valid = (
            components[nodes[:num_training_targets]]
            == components[nodes[num_training_targets + query_index]]
        ) & (distances < 2 * (steps + 1) - tol)
        candidates = sorted(
            ((float(distances[i]), names[i]) for i in np.flatnonzero(valid)),
            key=lambda x: (x[0], x[1]),
        )
        selected = candidates[:k]
        predictions.append(
            train_effects.loc[[name for _, name in selected]].mean(axis=0)
            if selected
            else fallback
        )
        # 3. Record graph coverage and the actual selected neighbors.
        diagnostics.append(
            {
                "target": target,
                "eligible_neighbors": len(candidates),
                "neighbors_used": len(selected),
                "fallback": not bool(selected),
                "component": int(components[nodes[num_training_targets + query_index]]),
                "weighted_degree": float(
                    adjacency[nodes[num_training_targets + query_index]].sum()
                ),
            }
        )
        for rank, (distance, name) in enumerate(selected, 1):
            neighbors.append(
                {
                    "target": target,
                    "rank": rank,
                    "neighbor": name,
                    "dsd": distance,
                    "weight": 1 / len(selected),
                }
            )
    return (
        pd.DataFrame(predictions, index=pd.Index(query, name="target")),
        pd.DataFrame(diagnostics).set_index("target"),
        pd.DataFrame(
            neighbors, columns=["target", "rank", "neighbor", "dsd", "weight"]
        ),
    )
