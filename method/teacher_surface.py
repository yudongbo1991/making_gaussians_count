"""Teacher surface for the final analytic method."""

import numpy as np


class TeacherSurface:

    def __init__(self, path):
        from plyfile import PlyData
        from scipy.spatial import cKDTree
        from scipy.special import expit

        vertex = PlyData.read(path)["vertex"]
        self.xyz = np.stack([vertex[k] for k in ("x", "y", "z")], axis=1).astype(np.float64)
        q = np.stack([vertex[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64)
        if len(self.xyz) < 5 or not np.isfinite(self.xyz).all() or (not np.isfinite(q).all()):
            raise ValueError("Finite teacher with at least five surface points required")
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        if (norm <= 0).any():
            raise ValueError("Invalid teacher rotation")
        q /= norm
        w, x, y, z = q.T
        self.normal = np.stack(
            [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=1
        )
        opacity = np.asarray(vertex["opacity"], dtype=np.float64)
        if not np.isfinite(opacity).all():
            raise ValueError("Invalid teacher opacity")
        self.opacity = expit(opacity)
        self.tree = cKDTree(self.xyz)

    def correspond(self, centers):
        centers = np.asarray(centers, dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1] != 3 or (not np.isfinite(centers).all()):
            raise ValueError("Finite three-dimensional centers required")
        _, nearest = self.tree.query(centers, workers=4)
        if not len(centers):
            return (nearest, np.empty(0))
        distinct = np.unique(nearest)
        local, _ = self.tree.query(self.xyz[distinct], k=5, workers=4)
        spacing = np.median(local[:, 1:], axis=1)[np.searchsorted(distinct, nearest)]
        spacing = spacing.clip(1e-12)
        plane = np.abs(((centers - self.xyz[nearest]) * self.normal[nearest]).sum(axis=1))
        reliability = self.opacity[nearest] * np.exp(-np.square(plane / spacing))
        return (nearest, reliability)
