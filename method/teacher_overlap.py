"""Teacher overlap for the final analytic method."""

import numpy as np


def ellipse_incidence(surface, centers):
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or (not np.isfinite(centers).all()):
        raise ValueError("Finite student centers required")
    _, ids = surface.tree.query(centers, k=5, workers=4)
    delta = centers[:, None, :] - surface.xyz[ids]
    tangent = ((delta * surface.u[ids]).sum(-1) / surface.scales[ids, 0]) ** 2
    tangent += ((delta * surface.v[ids]).sum(-1) / surface.scales[ids, 1]) ** 2
    plane = ((delta * surface.normal[ids]).sum(-1) / surface.spacing[ids]) ** 2
    opacity = surface.opacity[ids]
    log_opacity = np.full_like(opacity, -np.inf)
    np.log(opacity, out=log_opacity, where=opacity > 0)
    return (ids, np.exp(log_opacity - 0.5 * tangent - plane))
