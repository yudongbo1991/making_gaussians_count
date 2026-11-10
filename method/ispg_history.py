"""Late recovery for the final analytic method."""

import hashlib
import json
from pathlib import Path
import time
import torch
import numpy as np
from method.ispg import apply


@torch.enable_grad()
def contract_adaptive(model, cameras, pipe, background, mode, teacher, iteration, output):
    import gaussian_renderer as renderer

    path = Path(output)
    if path.exists():
        raise FileExistsError("Never overwrite a recovery decision")
    if mode != "geometry" or iteration not in (25000, 29000):
        raise ValueError("Declared late geometry protocol required")
    start = time.time()
    keep = model.slot_keep.detach().clone()
    n, k = keep.shape
    cameras = sorted(cameras, key=lambda c: c.image_name)
    cpu_rng = torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state().clone()
    observed = torch.zeros(n * k, device=keep.device, dtype=torch.int32)
    mass = torch.zeros(n * k, device=keep.device, dtype=torch.float64)
    capture = {}
    original = renderer.generate_neural_gaussians

    def generate(camera, pc, visible_mask=None, is_training=False):
        with torch.no_grad():
            values = list(original(camera, pc, visible_mask, is_training=True))
        parents = visible_mask.nonzero().flatten()
        capture["slot"] = (
            parents[:, None] * pc.n_offsets + torch.arange(pc.n_offsets, device=keep.device)
        ).flatten()[values[-1]]
        values[1] = values[1].detach().requires_grad_(True)
        capture["color"] = values[1]
        return tuple(values) if is_training else tuple(values[:5])

    renderer.generate_neural_gaussians = generate
    try:
        for camera in cameras:
            visible = renderer.prefilter_voxel(camera, model, pipe, background)
            result = renderer.render(camera, model, pipe, background, visible_mask=visible)
            rgb = result["render"]
            alpha = (
                torch.autograd.grad(
                    rgb,
                    capture["color"],
                    grad_outputs=torch.ones_like(rgb) / (rgb.shape[1] * rgb.shape[2]),
                )[0]
                .detach()
                .mean(1)
            )
            if not torch.isfinite(alpha).all() or alpha.min() < -1e-08:
                raise ValueError("Invalid compositing mass")
            slot = capture["slot"]
            observed.index_add_(0, slot, torch.ones_like(slot, dtype=torch.int32))
            mass.index_add_(0, slot, alpha.double().clamp_min(0))
            capture.clear()
            del result, rgb, alpha, visible
    finally:
        renderer.generate_neural_gaussians = original
    history = getattr(model, "ordinary_activation_history", None)
    if history is None or history.active.shape != keep.shape:
        raise ValueError("Actual ordinary activation history required")
    meta = history.metadata()
    if history.training_images != frozenset((camera.image_name for camera in cameras)):
        raise ValueError("History camera identity differs")
    if not meta["all_training_images_seen"]:
        raise ValueError("Every training image must be observed before history recovery")
    current_observed = observed.clone()
    historical = history.active.flatten()
    observed = torch.maximum(observed, historical.to(observed.dtype))
    witness = path.with_suffix(".ordinary_history.npz")
    np.savez_compressed(
        witness,
        current_observed=current_observed.cpu().numpy(),
        historical_active=history.active.cpu().numpy(),
        combined_observed=observed.cpu().numpy(),
    )
    meta.update(
        witness=str(witness),
        witness_sha256=hashlib.sha256(witness.read_bytes()).hexdigest(),
        historical_active_slots=int(historical.sum()),
        history_only_protected_slots=int(
            (historical & (current_observed == 0) & keep.flatten()).sum()
        ),
        active_history_tensor_bytes=history.active.numel() * history.active.element_size(),
    )
    try:
        after, proof = apply(model, cameras, pipe, background, keep, observed, mass, teacher, path)
    finally:
        model.slot_keep = keep
    if not torch.equal(after.any(1), keep.any(1)) or (after & ~keep).any():
        raise ValueError("Recovery changed anchor support or revived a slot")
    if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(
        gpu_rng, torch.cuda.get_rng_state()
    ):
        raise ValueError("Recovery changed global RNG")
    count = int((keep & ~after).sum())
    rows = [
        dict(image=row["image"], active=row["after_actual_gaussians"])
        for row in proof["training_views"]
    ]
    report = dict(
        iteration=iteration,
        mode=mode,
        accepted_count=count,
        late_policy="persistent_ordinary_training_and_current_all_training_inactivity_with_teacher_support",
        ordinary_training_history=meta,
        observation_scope="Union of actual ordinary neural-positive slots since15k core and current all-training generated neural-positive slots, including zero-radius or zero-mass slots",
        inactive_geometry=proof,
        admission_proposed=dict(per_view=rows),
        training_view_count=len(cameras),
        quantity_target=None,
        fixed_retirement_fraction=None,
        optimizer_steps=0,
        official_test_used=False,
        finite_training_rgb_radii_exact=True,
        changed_parameter_rows=False,
        changed_optimizer_rows=False,
        wall_seconds=time.time() - start,
        source_sha256={
            str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        },
    )
    with path.open("x") as stream:
        json.dump(report, stream, indent=2)
    model.slot_keep = after
    print("Late inactive geometry slots retired:", count, flush=True)
    return report
