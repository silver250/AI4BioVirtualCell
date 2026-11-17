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
    # Controls are cells by genes; effect is one predicted value per gene.
    control_array = np.asarray(controls, dtype=np.float64)
    effect_array = np.asarray(effect, dtype=np.float64)
    activation_random, donor_random = np.asarray(uniforms), np.asarray(donor_uniforms)
    if (
        control_array.ndim != 2
        or effect_array.shape != (control_array.shape[1],)
        or activation_random.shape != control_array.shape
        or donor_random.shape != control_array.shape
    ):
        raise ValueError("control, effect, and random array shapes do not match")
    if (
        not np.isfinite(control_array).all()
        or not np.isfinite(effect_array).all()
        or (control_array < 0).any()
        or (activation_random < 0).any()
        or (activation_random >= 1).any()
        or (donor_random < 0).any()
        or (donor_random >= 1).any()
    ):
        raise ValueError("expected finite nonnegative controls and uniforms in [0, 1)")
    # This is the mean expression we would like the predicted cells to have.
    intended = control_array.mean(axis=0) + effect_array
    target = np.maximum(intended, 0)
    # Add the same effect to every cell, clipping impossible negative values.
    if method == "additive":
        raw = control_array + effect_array
        predicted_cells = np.maximum(raw, 0)
        clipped = float((raw < 0).mean())
    # Alternative sensitivity model: sample positive values into some zeros.
    elif method == "hurdle":
        means = control_array.mean(axis=0)
        positive = control_array > 0
        counts = positive.sum(axis=0)
        positive_mean = np.divide(
            means * control_array.shape[0],
            counts,
            out=np.zeros_like(means),
            where=counts > 0,
        )
        zeros_fraction = 1 - counts / control_array.shape[0]
        increase = np.maximum(target - means, 0)
        cap = zeros_fraction * positive_mean
        activation_probability = np.divide(
            increase, cap, out=np.zeros_like(increase), where=cap > 0
        )
        activation_probability = np.clip(activation_probability, 0, 1)
        # Draw replacement values from cells where this gene was observed.
        ordered = np.sort(control_array, axis=0)
        donor_index = (
            control_array.shape[0]
            - counts
            + np.floor(donor_random * counts).astype(int)
        )
        donor_index = np.minimum(donor_index, control_array.shape[0] - 1)
        donor = np.take_along_axis(ordered, donor_index, axis=0)
        predicted_cells = np.where(
            (~positive) & (activation_random < activation_probability),
            donor,
            control_array,
        )
        expected = means + activation_probability * cap
        factor = np.divide(
            target, expected, out=np.ones_like(target), where=expected > 0
        )
        predicted_cells *= factor
        clipped = 0.0
    else:
        raise ValueError("method must be additive or hurdle")
    # Report how much the constructed cells differ from the intended response.
    return predicted_cells.astype(np.float32), {
        "fraction_negative_before_clip": clipped,
        "mean_absolute_effect_distortion": float(
            np.abs(
                predicted_cells.mean(axis=0) - control_array.mean(axis=0) - effect_array
            ).mean()
        ),
        "unreachable_fraction_genes": float(
            ((control_array.max(axis=0) == 0) & (target > 0)).mean()
        ),
        "fraction_negative_desired_mean": float((intended < 0).mean()),
        "fraction_positive": float((predicted_cells > 0).mean()),
    }
