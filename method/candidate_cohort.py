"""Candidate cohort for the final analytic method."""

import numpy as np
from method.lorenz_cohort import lorenz_candidate_count


def candidate_indices(response):
    response = np.asarray(response, dtype=np.float64)
    if response.ndim != 1 or not np.isfinite(response).all() or (response < 0).any():
        raise ValueError("Finite nonnegative measured response required")
    upper = lorenz_candidate_count(response)
    order = np.argsort(response, kind="stable")[:upper]
    return (
        order,
        dict(
            scope="Measured-response Lorenz tail before geometry, not geometry-adjusted global tail",
            measured_response_rows=len(response),
            candidate_rows=len(order),
            fixed_fraction=None,
            quantity_target=None,
            maximum_candidate_response=float(response[order].max()) if len(order) else None,
        ),
    )
