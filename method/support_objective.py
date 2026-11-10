"""Support objective for the final analytic method."""

import numpy as np
from method.support_statistics import prepare as original_prepare


def prepare(
    cost,
    group,
    mass,
    confidence,
    support_group,
    support_mass,
    support_confidence,
    target_ids,
    transition,
):
    data = original_prepare(
        cost, group, mass, confidence, support_group, support_mass, support_confidence
    )
    target_ids = np.asarray(target_ids)
    w = np.asarray(transition, dtype=np.float64)
    if target_ids.ndim != 2 or target_ids.shape != w.shape or len(w) != len(data["group_names"]):
        raise ValueError("One transition row per sorted original group required")
    if not np.issubdtype(target_ids.dtype, np.integer) or (target_ids < 0).any():
        raise ValueError("Nonnegative integral target identities required")
    if (
        not np.isfinite(w).all()
        or (w < 0).any()
        or (not np.allclose(w.sum(1), 1, rtol=0, atol=2e-12))
    ):
        raise ValueError("Finite nonnegative row-stochastic transition required")
    if w.shape[1] > 1 and (np.diff(np.sort(target_ids, axis=1), axis=1) == 0).any():
        raise ValueError("Distinct target identities per source row required")
    names, inverse = np.unique(target_ids, return_inverse=True)
    t = inverse.reshape(w.shape)
    trusted = np.bincount(
        t.ravel(), weights=(w * data["trusted_mass"][:, None]).ravel(), minlength=len(names)
    )
    budget = np.bincount(
        t.ravel(), weights=(w * data["weight"][:, None]).ravel(), minlength=len(names)
    )
    logedge = np.full_like(w, -np.inf)
    mask = (w > 0) & (data["trusted_mass"][:, None] > 0)
    source_log = np.full(len(w), -np.inf)
    np.log(data["trusted_mass"], out=source_log, where=data["trusted_mass"] > 0)
    logw = np.full_like(w, -np.inf)
    np.log(w, out=logw, where=w > 0)
    logedge[mask] = (logw + source_log[:, None])[mask]
    maximum = np.full(len(names), -np.inf)
    np.maximum.at(maximum, t.ravel(), logedge.ravel())
    edge = np.zeros_like(w)
    valid = np.isfinite(logedge)
    edge[valid] = np.exp(logedge[valid] - maximum[t[valid]])
    denominator = np.bincount(t.ravel(), weights=edge.ravel(), minlength=len(names))
    fractions = np.divide(edge, denominator[t], out=np.zeros_like(edge), where=denominator[t] > 0)
    assert np.isfinite(fractions).all() and ((fractions >= 0) & (fractions <= 1)).all()
    assert np.isclose(budget.sum(), data["weight"].sum(), rtol=2e-12, atol=1e-12)
    assert not ((denominator == 0) & (budget != 0)).any()
    logtrusted = np.full(len(names), -np.inf)
    positive = denominator > 0
    logtrusted[positive] = maximum[positive] + np.log(denominator[positive])
    data.update(
        target_names=names,
        target=t,
        transition=w,
        target_trusted_mass=trusted,
        target_budget=budget,
        group_fraction=fractions,
        target_log_trusted_mass=logtrusted,
    )
    return data
