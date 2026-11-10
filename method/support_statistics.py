"""Support statistics for the final analytic method."""

import numpy as np


def prepare(cost, group, mass, confidence, support_group, support_mass, support_confidence):
    risk = np.asarray(cost, dtype=np.float64)
    group = np.asarray(group)
    mass = np.asarray(mass, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    sg = np.asarray(support_group)
    sm = np.asarray(support_mass, dtype=np.float64)
    sq = np.asarray(support_confidence, dtype=np.float64)
    if risk.ndim != 1 or any((x.shape != risk.shape for x in [group, mass, confidence])):
        raise ValueError("Aligned candidate risk, identity, mass and confidence required")
    if sg.ndim != 1 or sm.shape != sg.shape or sq.shape != sg.shape:
        raise ValueError("Aligned full-support arrays required")
    if any((not np.issubdtype(x.dtype, np.integer) or (x < 0).any() for x in [group, sg])):
        raise ValueError("Nonnegative integral teacher identities required")
    if not all((np.isfinite(x).all() for x in [risk, mass, confidence, sm, sq])):
        raise ValueError("Finite input arrays required")
    if (
        (mass < 0).any()
        or (sm < 0).any()
        or any((((x < 0) | (x > 1)).any() for x in [confidence, sq]))
    ):
        raise ValueError("Nonnegative masses and confidence in [0,1] required")
    names, inverse = np.unique(np.concatenate([group, sg]), return_inverse=True)
    gi, si = (inverse[: len(group)], inverse[len(group) :])
    total = np.bincount(si, weights=sm, minlength=len(names))
    trusted = np.bincount(si, weights=sm * sq, minlength=len(names))
    b = mass * confidence
    proposed = np.bincount(gi, weights=mass, minlength=len(names))
    proposed_trusted = np.bincount(gi, weights=b, minlength=len(names))
    if (proposed > total + 1e-12 * np.maximum(total, np.finfo(float).tiny)).any():
        raise ValueError("Full support must contain all candidate raw mass")
    if (proposed_trusted > trusted + 1e-12 * np.maximum(trusted, np.finfo(float).tiny)).any():
        raise ValueError("Full support must contain all candidate confidence-weighted mass")
    weight = np.divide(trusted, total, out=np.zeros_like(total), where=total > 0)
    share = np.divide(b, trusted[gi], out=np.zeros_like(b), where=trusted[gi] > 0)
    scale = float(np.median(np.abs(risk[risk != 0]))) if (risk != 0).any() else 0.0
    return dict(
        risk=risk,
        group=gi,
        share=share,
        weight=weight,
        total_mass=total,
        trusted_mass=trusted,
        candidate_trusted_mass=b,
        group_names=names,
        candidate_confidence=confidence,
        candidate_mass=mass,
        risk_scale=scale,
    )
