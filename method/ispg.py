"""Inactive recovery for the final analytic method."""

import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from method.inactive_proposal import proposed_retirements
from method.teacher_ellipse import EllipticalTeacherSurface


@torch.no_grad()
def apply(model, cameras, pipe, background, keep, observed, mass, teacher, output):
    import gaussian_renderer as renderer

    start = time.time()
    n, k = keep.shape
    obs = observed.reshape(n, k).cpu().numpy()
    current = keep.cpu().numpy()
    measured = mass.reshape(n, k).cpu().numpy()
    ids = keep.flatten().nonzero().flatten()
    parent = ids // k
    offset = ids % k
    centers = (
        (model.get_anchor[parent] + model._offset[parent, offset] * model.get_scaling[parent, :3])
        .cpu()
        .numpy()
    )
    surface = EllipticalTeacherSurface(teacher)
    selected, confidence = surface.correspond(centers)
    group = np.zeros((n, k), dtype=np.int32)
    weight = np.zeros((n, k), dtype=np.float64)
    flat_ids = ids.cpu().numpy()
    group.flat[flat_ids] = selected
    weight.flat[flat_ids] = confidence
    retired, audit = proposed_retirements(obs, current, group, weight, measured)
    after = keep & ~torch.as_tensor(retired, device=keep.device)
    if not torch.equal(after.any(1), keep.any(1)):
        raise ValueError("Inactive proposal changed anchor survival")
    rows = []
    try:
        for camera in cameras:
            visible = renderer.prefilter_voxel(camera, model, pipe, background)
            model.slot_keep = keep
            before = renderer.render(camera, model, pipe, background, visible_mask=visible)
            model.slot_keep = after
            result = renderer.render(camera, model, pipe, background, visible_mask=visible)
            rgb_equal = torch.equal(before["render"], result["render"])
            radii_equal = torch.equal(before["radii"], result["radii"])
            rows.append(
                dict(
                    image=camera.image_name,
                    rgb_exact=rgb_equal,
                    radii_exact=radii_equal,
                    before_actual_gaussians=int((before["radii"] > 0).sum()),
                    after_actual_gaussians=int((result["radii"] > 0).sum()),
                    max_rgb_difference=float((before["render"] - result["render"]).abs().max()),
                )
            )
            del before, result, visible
    finally:
        model.slot_keep = None
    passed = bool(rows) and all((r["rgb_exact"] and r["radii_exact"] for r in rows))
    witness = Path(output).with_suffix(".inactive_slots.npz")
    np.savez_compressed(
        witness,
        observed=obs,
        keep_before=current,
        teacher_group=group,
        teacher_confidence=weight,
        mass=measured,
        retired=retired,
        keep_after=after.cpu().numpy(),
    )
    audit.update(
        status=(
            "finite_training_rgb_and_radii_exact"
            if passed
            else "failed_finite_render_gate_preserved"
        ),
        finite_render_invariance_verified=passed,
        training_views=rows,
        training_view_count=len(rows),
        witness=str(witness),
        witness_sha256=hashlib.sha256(witness.read_bytes()).hexdigest(),
        teacher_sha256=hashlib.sha256(Path(teacher).read_bytes()).hexdigest(),
        wall_seconds=time.time() - start,
        official_test_used=False,
        optimizer_steps=0,
        geometry_correspondence="Five-neighbour learned teacher ellipse score; proxy, not exact coverage",
    )
    if not passed:
        import json

        witness.with_suffix(".failed.json").write_text(json.dumps(audit, indent=2))
        raise ValueError(
            "Inactive retirement changed finite training RGB/radii; evidence preserved"
        )
    return (after, audit)
