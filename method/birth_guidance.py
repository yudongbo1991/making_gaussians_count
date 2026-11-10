"""Birth guidance for the final analytic method."""

import time
import numpy as np
import torch
from method.birth_surface import Guidance as CachedSurface
from method.birth_offsets import supported_offsets
from method.constants import KERNEL_ALPHA_CUTOFF


class Guidance(CachedSurface):

    def initialize_normalized(self, position, scaling, cell_size, n_offsets):
        start = time.perf_counter()
        centers = position.detach().cpu().numpy()
        group, confidence = self.surface.correspond(centers)
        surface = self.surface
        delta, audit = supported_offsets(
            centers,
            surface.xyz[group],
            surface.u[group],
            surface.v[group],
            surface.normal[group],
            surface.scales[group],
            surface.spacing[group],
            surface.opacity[group],
            cell_size,
            n_offsets,
        )
        normalized = torch.as_tensor(delta, device=position.device) / cell_size
        reliable = confidence >= KERNEL_ALPHA_CUTOFF
        numerical_halvings = 0

        def measured(world):
            displacement = world - surface.xyz[group, None, :]
            tangent = (
                (displacement * surface.u[group, None, :]).sum(2) / surface.scales[group, 0, None]
            ) ** 2
            tangent += (
                (displacement * surface.v[group, None, :]).sum(2) / surface.scales[group, 1, None]
            ) ** 2
            plane = (
                (displacement * surface.normal[group, None, :]).sum(2)
                / surface.spacing[group, None]
            ) ** 2
            return surface.opacity[group, None] * np.exp(-0.5 * tangent - plane)

        with torch.no_grad():
            extent = scaling[:, :3].exp()
            while True:
                world = (
                    (position[:, None, :] + normalized * extent[:, None, :])
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                measured_confidence = measured(world)
                bad = reliable & (measured_confidence.min(1) < KERNEL_ALPHA_CUTOFF)
                if not bad.any():
                    break
                if np.any(bad & ~(normalized.abs().amax((1, 2)).cpu().numpy() > 0)):
                    raise ValueError("Supported original center became unsupported at zero offset")
                normalized[torch.as_tensor(bad, device=position.device)] *= 0.5
                numerical_halvings += int(bad.sum())
        audit.update(
            reliable_anchor_count=int(reliable.sum()),
            unknown_anchor_count=int((~reliable).sum()),
            teacher_ply=str(self.path),
            teacher_sha256=self.sha256,
            born_anchor_count=len(centers),
            cell_size=float(cell_size),
            guidance_wall_seconds=time.perf_counter() - start,
            actual_float32_selected_teacher_support_verified=True,
            actual_normalized_offset_numerical_halvings=numerical_halvings,
        )
        return (normalized, audit)
