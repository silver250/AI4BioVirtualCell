"""Two explicit assumptions for making predicted cells from control cells.

All values are log1p(total-count normalized), not raw counts.
"""
import numpy as np


def generate_cells(controls, effect, uniforms, donor_uniforms, method):
    """Return predicted profiles and diagnostics, using supplied random numbers.

    Hurdle: decrease existing positive values proportionally; for increases,
    activate a fraction of zero entries using empirical positive control values.
    If a gene is never observed positive among sampled controls it cannot be
    activated by this method. This is a provisional population model, not a
    learned perturbation distribution.
    """
    c = np.asarray(controls, dtype=np.float64)
    d = np.asarray(effect, dtype=np.float64)
    u, v = np.asarray(uniforms), np.asarray(donor_uniforms)
    if c.ndim != 2 or d.shape != (c.shape[1],) or u.shape != c.shape or v.shape != c.shape:
        raise ValueError('control, effect, and random array shapes do not match')
    if (not np.isfinite(c).all() or not np.isfinite(d).all() or (c < 0).any()
            or (u < 0).any() or (u >= 1).any() or (v < 0).any() or (v >= 1).any()):
        raise ValueError('expected finite nonnegative controls and uniforms in [0, 1)')
    intended = c.mean(axis=0) + d
    target = np.maximum(intended, 0)
    if method == 'additive':
        raw = c + d
        out = np.maximum(raw, 0)
        clipped = float((raw < 0).mean())
    elif method == 'hurdle':
        means = c.mean(axis=0)
        positive = c > 0
        counts = positive.sum(axis=0)
        pos_mean = np.divide(means*c.shape[0], counts, out=np.zeros_like(means), where=counts > 0)
        zeros_fraction = 1 - counts/c.shape[0]
        increase = np.maximum(target-means, 0)
        cap = zeros_fraction * pos_mean
        probs = np.divide(increase, cap, out=np.zeros_like(increase), where=cap > 0)
        probs = np.clip(probs, 0, 1)
        ordered = np.sort(c, axis=0)
        donor_index = c.shape[0] - counts + np.floor(v * counts).astype(int)
        donor_index = np.minimum(donor_index, c.shape[0]-1)
        donor = np.take_along_axis(ordered, donor_index, axis=0)
        out = np.where((~positive) & (u < probs), donor, c)
        expected = means + probs*cap
        factor = np.divide(target, expected, out=np.ones_like(target), where=expected > 0)
        out *= factor
        clipped = 0.0
    else:
        raise ValueError('method must be additive or hurdle')
    return out.astype(np.float32), {
        'fraction_negative_before_clip': clipped,
        'mean_absolute_effect_distortion': float(np.abs(out.mean(axis=0)-c.mean(axis=0)-d).mean()),
        'unreachable_fraction_genes': float(((c.max(axis=0) == 0) & (target > 0)).mean()),
        'fraction_negative_desired_mean': float((intended < 0).mean()),
        'fraction_positive': float((out > 0).mean()),
    }
