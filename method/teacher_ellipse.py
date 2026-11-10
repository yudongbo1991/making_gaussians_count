"""Teacher ellipse for the final analytic method."""

import numpy as np
from method.teacher_surface import TeacherSurface


def select_ellipse(delta, tangent_u, tangent_v, normal, scales, spacing, opacity):
    delta = np.asarray(delta, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    opacity = np.asarray(opacity, dtype=np.float64)
    if delta.ndim != 3 or delta.shape[-1] != 3 or scales.shape != delta.shape[:-1] + (2,):
        raise ValueError("Aligned neighbour ellipses required")
    for value in (delta, tangent_u, tangent_v, normal, scales, spacing, opacity):
        if not np.isfinite(value).all():
            raise ValueError("Finite geometry required")
    if (scales <= 0).any() or (spacing <= 0).any() or ((opacity < 0) | (opacity > 1)).any():
        raise ValueError("Positive scales/spacing and activated opacity required")
    tangent = ((delta * tangent_u).sum(-1) / scales[..., 0]) ** 2
    tangent += ((delta * tangent_v).sum(-1) / scales[..., 1]) ** 2
    plane = ((delta * normal).sum(-1) / spacing) ** 2
    log_opacity = np.full_like(opacity, -np.inf)
    np.log(opacity, out=log_opacity, where=opacity > 0)
    score = log_opacity - 0.5 * tangent - plane
    chosen = np.argmax(score, axis=1)
    return (chosen, np.exp(score[np.arange(len(chosen)), chosen]))


class EllipticalTeacherSurface(TeacherSurface):

    def __init__(self, path):
        super().__init__(path)
        from plyfile import PlyData

        vertex = PlyData.read(path)["vertex"]
        self.scales = np.exp(
            np.stack([vertex[f"scale_{i}"] for i in range(2)], axis=1).astype(np.float64)
        )
        q = np.stack([vertex[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        w, x, y, z = q.T
        self.u = np.stack(
            [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=1
        )
        self.v = np.stack(
            [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], axis=1
        )
        local, _ = self.tree.query(self.xyz, k=5, workers=4)
        self.spacing = np.median(local[:, 1:], axis=1).clip(1e-12)
        if not np.isfinite(self.scales).all() or (self.scales <= 0).any():
            raise ValueError("Invalid learned tangent scales")

    def correspond(self, centers):
        centers = np.asarray(centers, dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1] != 3 or (not np.isfinite(centers).all()):
            raise ValueError("Finite 3D centers required")
        nearest = np.empty(len(centers), dtype=np.int64)
        reliability = np.empty(len(centers), dtype=np.float64)
        for start in range(0, len(centers), 32768):
            chunk = centers[start : start + 32768]
            _, ids = self.tree.query(chunk, k=5, workers=4)
            chosen, weight = select_ellipse(
                chunk[:, None, :] - self.xyz[ids],
                self.u[ids],
                self.v[ids],
                self.normal[ids],
                self.scales[ids],
                self.spacing[ids],
                self.opacity[ids],
            )
            nearest[start : start + len(chunk)] = ids[np.arange(len(chunk)), chosen]
            reliability[start : start + len(chunk)] = weight
        return (nearest, reliability)
