"""Birth offsets for the final analytic method."""

import numpy as np
from method.constants import KERNEL_ALPHA_CUTOFF


def supported_offsets(
    centers, teacher_centers, u, v, normal, scales, spacing, opacity, cell_size, n_offsets=10
):
    arrays = [
        np.asarray(x, dtype=np.float64)
        for x in (centers, teacher_centers, u, v, normal, scales, spacing, opacity)
    ]
    centers, teacher_centers, u, v, normal, scales, spacing, opacity = arrays
    n = len(centers)
    if any((x.shape != (n, 3) for x in (centers, teacher_centers, u, v, normal))):
        raise ValueError("Aligned center and tangent frames required")
    if scales.shape != (n, 2) or spacing.shape != (n,) or opacity.shape != (n,):
        raise ValueError("Aligned learned surface scales required")
    if (
        any((not np.isfinite(x).all() for x in arrays))
        or (scales <= 0).any()
        or (spacing <= 0).any()
    ):
        raise ValueError("Finite positive geometry required")
    if ((opacity < 0) | (opacity > 1)).any() or not np.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("Valid activated opacity and cell width required")
    if (
        not isinstance(n_offsets, int)
        or isinstance(n_offsets, bool)
        or n_offsets < 2
        or n_offsets % 2
    ):
        raise ValueError("Existing even slot count required")
    basis = np.stack((u, v, normal), axis=1)
    if not np.allclose(basis @ basis.transpose(0, 2, 1), np.eye(3), atol=1e-06, rtol=0):
        raise ValueError("Orthonormal teacher frame required")
    centers = centers.astype(np.float32).astype(np.float64)
    if not np.isfinite(centers).all():
        raise ValueError("Centers must be representable by ordinary float32 growth")
    displacement = centers - teacher_centers
    tangent = np.stack(
        ((displacement * u).sum(1) / scales[:, 0], (displacement * v).sum(1) / scales[:, 1]), axis=1
    )
    plane = ((displacement * normal).sum(1) / spacing) ** 2
    log_opacity = np.full(n, -np.inf)
    np.log(opacity, out=log_opacity, where=opacity > 0)
    tangent_squared = (tangent**2).sum(1)
    log_confidence = log_opacity - 0.5 * tangent_squared - plane
    reliable = log_confidence >= np.log(KERNEL_ALPHA_CUTOFF)
    margin = np.maximum(
        2 * (log_opacity - np.log(KERNEL_ALPHA_CUTOFF) - plane) - tangent_squared, 0
    )
    margin[~reliable] = 0
    angles = np.arange(n_offsets // 2) * np.pi / (n_offsets // 2)
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    projection = np.abs(tangent @ directions.T).max(1)
    denominator = np.sqrt(projection**2 + margin) + projection
    allowance = np.divide(margin, denominator, out=np.zeros(n), where=denominator > 0)
    raw = (
        u[:, None, :] * scales[:, 0, None, None] * np.cos(angles)[None, :, None]
        + v[:, None, :] * scales[:, 1, None, None] * np.sin(angles)[None, :, None]
    )
    raw = np.concatenate((raw, -raw), axis=1)
    cell_allowance = cell_size / (2 * np.abs(raw).max((1, 2)))
    radius = np.minimum(1, np.minimum(allowance, cell_allowance))
    radius[~reliable] = 0

    def confidence_at(points):
        d = points - teacher_centers[:, None, :]
        t = ((d * u[:, None, :]).sum(2) / scales[:, 0, None]) ** 2
        t += ((d * v[:, None, :]).sum(2) / scales[:, 1, None]) ** 2
        p = ((d * normal[:, None, :]).sum(2) / spacing[:, None]) ** 2
        return np.exp(log_opacity[:, None] - 0.5 * t - p)

    halvings = np.zeros(n, dtype=np.int64)
    while True:
        delta = (raw * radius[:, None, None]).astype(np.float32)
        points = (centers.astype(np.float32)[:, None, :] + delta).astype(np.float64)
        bad = reliable & (confidence_at(points).min(1) < KERNEL_ALPHA_CUTOFF) & (radius > 0)
        if not bad.any():
            break
        radius[bad] *= 0.5
        halvings[bad] += 1
    return (
        delta,
        dict(
            reliable_anchors=int(reliable.sum()),
            unknown_anchors=int((~reliable).sum()),
            nonzero_guided_anchors=int((np.abs(delta).max((1, 2)) > 0).sum()),
            numerical_radius_halvings=int(halvings.sum()),
            radius_source="Learned tangent scales, per-anchor remaining teacher support, existing cell",
            teacher_kernel_cutoff=KERNEL_ALPHA_CUTOFF,
            radius_in_teacher_standard_deviations_quantiles=np.quantile(
                radius, [0, 0.25, 0.5, 0.75, 1]
            ).tolist(),
            added_anchors=0,
            added_slots=0,
            quantity_target=None,
            normal_bandwidth_is_spacing_proxy=True,
            future_quality_verified=False,
        ),
    )
