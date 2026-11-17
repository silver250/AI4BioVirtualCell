"""Simple gene-by-gene effects for held-out targets."""
import numpy as np
import pandas as pd


def predict_baselines(training_effects, query_targets):
    """Return a zero effect and mean training-target effect for every query."""
    if training_effects.empty or not training_effects.index.is_unique or not training_effects.columns.is_unique:
        raise ValueError('training effects must have unique targets and genes')
    if not np.isfinite(training_effects.to_numpy(dtype=float)).all():
        raise ValueError('training effects must be finite')
    queries=list(query_targets)
    if not queries or len(set(queries)) != len(queries) or set(queries) & set(training_effects.index):
        raise ValueError('query labels must be unique, nonempty and absent from training')
    shape=(len(queries),training_effects.shape[1])
    index=pd.Index(queries,name='target')
    zero=pd.DataFrame(np.zeros(shape),index=index,columns=training_effects.columns)
    mean=pd.DataFrame(np.tile(training_effects.mean(axis=0).to_numpy(),(len(queries),1)),index=index,columns=training_effects.columns)
    return {'no_change':zero,'mean_effect':mean}
