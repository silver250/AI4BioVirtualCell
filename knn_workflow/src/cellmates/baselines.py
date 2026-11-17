"""Simple gene-by-gene effects for held-out targets."""

import numpy as np
import pandas as pd


def predict_baselines(training_effects, query_targets):
    """Return no-change and mean-effect predictions for each unseen target.

    Rows of training_effects are known targets; columns are measured genes.
    The training mean gives every target equal weight, not every cell.
    """
    # Check the training table and prevent a known target being used as a query.
    if (
        training_effects.empty
        or not training_effects.index.is_unique
        or not training_effects.columns.is_unique
    ):
        raise ValueError("training effects must have unique targets and genes")
    if not np.isfinite(training_effects.to_numpy(dtype=float)).all():
        raise ValueError("training effects must be finite")
    queries = list(query_targets)
    if (
        not queries
        or len(set(queries)) != len(queries)
        or set(queries) & set(training_effects.index)
    ):
        raise ValueError(
            "query labels must be unique, nonempty and absent from training"
        )

    # Both predictions have one row per query and one column per measured gene.
    shape = (len(queries), training_effects.shape[1])
    index = pd.Index(queries, name="target")
    no_change = pd.DataFrame(
        np.zeros(shape), index=index, columns=training_effects.columns
    )
    training_mean = training_effects.mean(axis=0).to_numpy()
    mean_effect = pd.DataFrame(
        np.tile(training_mean, (len(queries), 1)),
        index=index,
        columns=training_effects.columns,
    )
    return {"no_change": no_change, "mean_effect": mean_effect}
