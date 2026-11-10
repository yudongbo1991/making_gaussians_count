"""Inactive proposal for the final analytic method."""

import numpy as np
from method.constants import KERNEL_ALPHA_CUTOFF


def proposed_retirements(observed, keep, teacher_group, teacher_confidence, mass):
    observed = np.asarray(observed)
    keep = np.asarray(keep)
    group = np.asarray(teacher_group)
    confidence = np.asarray(teacher_confidence, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    if observed.ndim != 2 or any(
        (value.shape != observed.shape for value in (keep, group, confidence, mass))
    ):
        raise ValueError("All current anchor/slot rows must align")
    if (
        keep.dtype != np.bool_
        or not np.issubdtype(observed.dtype, np.integer)
        or (not np.issubdtype(group.dtype, np.integer))
    ):
        raise ValueError("Boolean current mask and integral observation/group identities required")
    if (
        (observed < 0).any()
        or (group < 0).any()
        or (not np.isfinite(confidence).all())
        or (not np.isfinite(mass).all())
        or ((confidence < 0) | (confidence > 1)).any()
        or (mass < 0).any()
    ):
        raise ValueError("Finite nonnegative geometry/observations/mass required")
    if (mass[observed == 0] != 0).any():
        raise ValueError("Unobserved slots cannot have measured compositing mass")
    support = keep & (observed > 0) & (mass > 0) & (confidence >= KERNEL_ALPHA_CUTOFF)
    supported_groups = np.unique(group[support])
    parent_has_observed = (keep & (observed > 0)).any(axis=1)
    eligible = keep & (observed == 0) & parent_has_observed[:, None]
    covered = (confidence >= KERNEL_ALPHA_CUTOFF) & np.isin(group, supported_groups)
    retirement = eligible & covered
    audit = dict(
        status="pending_finite_render_check",
        current_slots=int(keep.sum()),
        unobserved_surviving_slots=int((keep & (observed == 0)).sum()),
        proposed_retired_slots=int(retirement.sum()),
        teacher_kernel_alpha_cutoff=KERNEL_ALPHA_CUTOFF,
        observed_slots_retired=int((retirement & (observed > 0)).sum()),
        support_groups=len(supported_groups),
        quantity_target=None,
        fixed_retirement_fraction=None,
        geometry_scope="Same selected teacher ellipse with positive measured surviving observed-slot support; proxy only",
        finite_render_invariance_verified=False,
    )
    return (retirement, audit)
