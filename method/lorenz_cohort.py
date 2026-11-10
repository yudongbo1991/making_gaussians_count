"""Lorenz cohort for the final analytic method."""

import numpy as np


def lorenz_candidate_count(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Finite nonnegative one-dimensional relevance required")
    n = len(values)
    if not n:
        return 0
    if not values.any():
        return n
    ordered = np.sort(values, kind="stable")
    if ordered[0] == ordered[-1]:
        return 0
    normalized = ordered / ordered[-1]
    mass = np.cumsum(normalized)
    mass /= mass[-1]
    row_share = np.arange(1, n + 1, dtype=np.float64) / n
    gap = row_share - mass
    index = int(np.argmax(gap[:-1]))
    return index + 1 if gap[index] > 0 else 0
