"""Explicit cell-eval2 configuration for the exploratory validation workflow."""
from dataclasses import asdict
from importlib.metadata import version
from cell_eval2 import EvalConfig, aggregate_metrics, compute_metrics
from cell_eval2.config import DEParams


def score_validation(predicted, observed, include_de=False):
    """Score aligned AnnData objects; callers must enforce the split boundary.

    Both sides are log1p total-count-normalized to 10,000 before this call.
    DE is opt-in while the cell population and zero-variance warnings are studied.
    Returns per-target results, metric means and the effective configuration.
    """
    if not predicted.var_names.equals(observed.var_names):
        raise ValueError('predicted and observed gene names/order must match')
    config = EvalConfig(metrics='vcc' if include_de else ['expr_mae','pds_l1'],
                        pert_col='target_gene', control='non-targeting',
                        input_type='lognorm', target_sum=10_000,
                        autodetect_input_type=False, control_source='real',
                        device='cpu', num_threads=1, de=DEParams(backend='scanpy'))
    scores=compute_metrics(predicted,observed,config=config)
    return scores,aggregate_metrics(scores),{'cell_eval2':version('cell-eval2'),'config':asdict(config)}
