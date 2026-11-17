"""Transparent one-hop KNN for target-by-gene perturbation effects.

Input edges must already be mapped to unambiguous human gene symbols and have
columns gene_a, gene_b, score. Scores are STRING confidence in [0, 1].
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class KNNResult:
    predictions: pd.DataFrame
    diagnostics: pd.DataFrame
    neighbors: pd.DataFrame


def predict_direct_knn(train_effects, query_targets, edges, k=5, min_score=0.4, weighting="equal"):
    """Predict each unseen target using only training effects.

    Rank undirected direct edges by confidence, break ties by gene symbol, and
    average up to k distinct neighbors, using equal or normalized score weights.
    With no neighbors use the training mean. Return coverage and neighbor weights.
    """
    if weighting not in ('equal', 'score'):
        raise ValueError('weighting must be equal or score')
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError('k must be a positive integer')
    if not 0 <= min_score <= 1:
        raise ValueError('min_score must be in [0, 1]')
    if train_effects.empty or not train_effects.index.is_unique or not train_effects.columns.is_unique:
        raise ValueError('training effects must be nonempty with unique targets and genes')
    if not all(isinstance(x, str) and x for x in train_effects.index):
        raise ValueError('training target labels must be nonempty strings')
    if not np.isfinite(train_effects.to_numpy(dtype=float)).all():
        raise ValueError('training effects must be finite numbers')
    query = list(query_targets)
    if not query or len(set(query)) != len(query) or any(not isinstance(x, str) or not x for x in query):
        raise ValueError('query targets must be unique nonempty strings')
    if set(query) & set(train_effects.index):
        raise ValueError('query and training targets must be disjoint')
    if not {'gene_a', 'gene_b', 'score'} <= set(edges.columns):
        raise ValueError('edges require gene_a, gene_b, score columns')
    network = edges[['gene_a', 'gene_b', 'score']].copy()
    if network[['gene_a', 'gene_b']].isna().any().any() or not all(
        isinstance(x, str) and x for x in network[['gene_a', 'gene_b']].to_numpy().ravel()
    ):
        raise ValueError('edge endpoints must be nonempty gene symbols')
    network['score'] = pd.to_numeric(network['score'], errors='raise')
    if not np.isfinite(network['score']).all() or not network['score'].between(0, 1).all():
        raise ValueError('edge scores must be finite and in [0, 1]')
    network = network[(network.score >= min_score) & (network.gene_a != network.gene_b)]
    reverse = network.rename(columns={'gene_a': 'gene_b', 'gene_b': 'gene_a'})
    network = pd.concat([network, reverse], ignore_index=True)
    network = network.groupby(['gene_a', 'gene_b'], as_index=False)['score'].max()
    candidates = network[network.gene_a.isin(query) & network.gene_b.isin(train_effects.index)]
    candidates = candidates.sort_values(['gene_a', 'score', 'gene_b'], ascending=[True, False, True])
    fallback = train_effects.mean(axis=0)
    predictions, diagnostics, chosen = [], [], []
    for target in query:
        possible = candidates[candidates.gene_a == target]
        selected = possible.head(k)
        genes = selected.gene_b.tolist()
        if genes:
            weights = np.ones(len(genes)) if weighting == 'equal' else selected.score.to_numpy(dtype=float)
            if weights.sum() <= 0:
                raise ValueError('selected weights must sum to a positive number')
            weights = weights / weights.sum()
            predictions.append(pd.Series(weights @ train_effects.loc[genes].to_numpy(), index=train_effects.columns))
        else:
            weights = []
            predictions.append(fallback)
        diagnostics.append({'target': target, 'eligible_neighbors': len(possible),
                            'neighbors_used': len(genes), 'fallback': not bool(genes)})
        for rank, row in enumerate(selected.itertuples(index=False), 1):
            chosen.append({'target': target, 'rank': rank, 'neighbor': row.gene_b, 'score': row.score, 'weight': weights[rank - 1]})
    return KNNResult(pd.DataFrame(predictions, index=pd.Index(query, name='target')),
                     pd.DataFrame(diagnostics).set_index('target'),
                     pd.DataFrame(chosen, columns=['target', 'rank', 'neighbor', 'score', 'weight']))
