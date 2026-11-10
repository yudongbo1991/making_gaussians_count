"""Risk discount for the final analytic method."""

import numpy as np


def discounted_risk(cost, nearest, reliability, support_nearest, support_mass):
    risk = np.asarray(cost, dtype=np.float64)
    g = np.asarray(nearest)
    q = np.asarray(reliability, dtype=np.float64)
    sg = np.asarray(support_nearest)
    sm = np.asarray(support_mass, dtype=np.float64)
    if (
        risk.ndim != 1
        or g.shape != risk.shape
        or q.shape != risk.shape
        or (sg.ndim != 1)
        or (sm.shape != sg.shape)
    ):
        raise ValueError("Aligned candidate and support arrays required")
    if any((not np.issubdtype(x.dtype, np.integer) or (x < 0).any() for x in [g, sg])):
        raise ValueError("Nonnegative integer group identities required")
    if (
        not all((np.isfinite(x).all() for x in [risk, q, sm]))
        or ((q < 0) | (q > 1)).any()
        or (sm < 0).any()
    ):
        raise ValueError("Finite risks, bounded reliability, and nonnegative mass required")
    names, inverse = np.unique(np.concatenate([g, sg]), return_inverse=True)
    cg = inverse[: len(g)]
    full = inverse[len(g) :]
    count = np.bincount(full[sm > 0], minlength=len(names))
    multiplicity = np.maximum(count[cg], 1)
    factor = 1 + q * np.log1p(multiplicity - 1)
    adjusted = np.where(risk < 0, risk * factor, risk / factor)
    return (
        adjusted,
        dict(
            discount_fixed_before_ranking=True,
            candidates_with_discount=int((factor > 1).sum()),
            median_factor=float(np.median(factor)) if len(factor) else 1.0,
        ),
    )
