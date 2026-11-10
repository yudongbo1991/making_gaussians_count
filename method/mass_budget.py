"""Mass budget for the final analytic method."""

import numpy as np


def reweight(data):
    logs = np.asarray(data["target_log_trusted_mass"], dtype=np.float64)
    base_budget = np.asarray(data["target_budget"], dtype=np.float64)
    if logs.shape != base_budget.shape or np.isnan(logs).any() or np.isposinf(logs).any():
        raise ValueError("Aligned finite-or-minus-infinity log trusted masses required")
    if not np.isfinite(base_budget).all() or (base_budget < 0).any():
        raise ValueError("Finite nonnegative base_budget budgets required")
    probability = np.zeros_like(logs)
    positive = np.isfinite(logs)
    total = float(base_budget.sum())
    if positive.any():
        probability[positive] = np.exp(logs[positive] - logs[positive].max())
        probability /= probability.sum()
    elif total != 0:
        raise ValueError("Nonzero prior budget requires positive trusted support")
    budget = total * probability
    assert np.isclose(budget.sum(), total, rtol=2e-12, atol=1e-12)
    result = dict(data)
    result.update(target_budget=budget, target_mass_probability=probability, total_budget=total)
    return result
