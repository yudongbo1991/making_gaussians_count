"""HPRG: rendering and geometric removal risks, followed by ISPG."""

from method.native import load_analyzers
import json
from pathlib import Path
import time
import numpy as np
import torch


@torch.enable_grad()
def contract(
    model,
    cameras,
    pipe,
    background,
    views,
    iteration,
    output,
    geometry_teacher="",
    source_path=None,
    resolution=-1,
):
    import gaussian_renderer as renderer
    from scene.adaptive_recovery import pareto_knee
    from method.candidate_cohort import candidate_indices
    from method.quality_profile import joint_profile
    from method.teacher_surface import TeacherSurface
    from scene.adaptive_recovery import teacher_contract
    from scene.optimizer_rows import retain_optimizer_rows
    from utils.loss_utils import ssim, l1_loss

    if views < 4 or iteration != 15000:
        raise ValueError("No fixed fractions; declared15k calibration required")
    if (
        model.slot_keep is not None
        or not model.get_color_mlp.training
        or torch.count_nonzero(background)
    ):
        raise ValueError("Fresh dense training model and black background required")
    if hasattr(model, "offset_gradient_squared_accum"):
        raise ValueError("Growth must have ended before row retirement")
    output = Path(output)
    if output.exists():
        raise ValueError("Preserve prior calibration report")
    cpu_rng = torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state().clone()
    import lpips

    perceptual = lpips.LPIPS(net="vgg").cuda().eval().requires_grad_(False)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(gpu_rng)
    fields = ("anchor", "offset", "anchor_feat", "scaling", "rotation", "opacity")
    snapshots = {key: getattr(model, "_" + key).detach().clone() for key in fields}
    cameras = sorted(cameras, key=lambda c: c.image_name)
    admission = [c for i, c in enumerate(cameras) if i % 8 == 0]
    fitting = [c for i, c in enumerate(cameras) if i % 8]
    fitting = [
        fitting[i] for i in np.linspace(0, len(fitting) - 1, min(views, len(fitting)), dtype=int)
    ]
    bg = background
    n = len(model.get_anchor)
    k = model.n_offsets
    teacher_provenance = teacher_contract(geometry_teacher, source_path, resolution)
    surface = TeacherSurface(geometry_teacher)
    from method.analytic_risk import moments as analytic_dense_moments

    analytic_modules = load_analyzers()
    curvature_samples = 0
    start = time.time()

    def measure_response(context):
        capture_start = time.time()
        anchor_error = torch.zeros(n, device="cuda")
        frustum = torch.zeros(n, device="cuda", dtype=torch.int32)
        anchor_mass = torch.zeros_like(anchor_error)
        slot_mass = torch.zeros(n * k, device="cuda")
        slot_error = torch.zeros(n * k, device="cuda")
        observed = torch.zeros(n * k, device="cuda", dtype=torch.int32)
        anchor_damage = torch.full((n,), -torch.inf, device="cuda")
        slot_damage = torch.full((n * k,), -torch.inf, device="cuda")
        original = renderer.generate_neural_gaussians
        captured = {}
        records = []

        def generate(camera, pc, visible_mask=None, is_training=False):
            with torch.no_grad():
                values = list(original(camera, pc, visible_mask, is_training=True))
            parents = visible_mask.nonzero().flatten()
            captured["slot"] = (parents[:, None] * k + torch.arange(k, device="cuda")).flatten()[
                values[-1]
            ]
            values[1] = values[1].detach().requires_grad_(True)
            captured["color"] = values[1]
            values[2] = values[2].detach().requires_grad_(True)
            captured["opacity"] = values[2]
            return tuple(values) if is_training else tuple(values[:5])

        renderer.generate_neural_gaussians = generate
        try:
            for c in cameras:
                with torch.no_grad():
                    visible = renderer.prefilter_voxel(c, model, pipe, bg)
                    frustum += visible.int()
                pkg = renderer.render(c, model, pipe, bg, visible_mask=visible)
                rgb = pkg["render"]
                error = (rgb.detach().clamp(0, 1) - c.original_image.to("cuda")).abs().mean(0)
                weights = error[None].expand(3, -1, -1) / error.numel()
                jac = torch.autograd.grad(
                    rgb, captured["color"], grad_outputs=weights, retain_graph=True
                )[0].detach()
                identity = float(
                    (
                        (captured["color"].detach() * jac).sum(0)
                        - (rgb.detach() * weights).sum((1, 2))
                    )
                    .abs()
                    .max()
                )
                if not torch.isfinite(jac).all() or jac.min() < -1e-08 or identity > 2e-05:
                    raise ValueError("Invalid compositing identity; no relaxed threshold")
                ids = captured["slot"]
                anchor_error.index_add_(0, ids // k, jac[:, 0])
                slot_error.index_add_(0, ids, jac[:, 0])
                observed.index_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))
                mass_weights = torch.ones_like(rgb) / (rgb.shape[1] * rgb.shape[2])
                alpha_mass = (
                    torch.autograd.grad(
                        rgb, captured["color"], grad_outputs=mass_weights, retain_graph=True
                    )[0]
                    .detach()
                    .double()
                )

                def mass_identity(rgb, mass_weights, alpha_mass):
                    return float(
                        (
                            (captured["color"].detach().double() * alpha_mass).sum(0)
                            - (rgb.detach().double() * mass_weights.double()).sum((1, 2))
                        )
                        .abs()
                        .max()
                    )

                mass_error = mass_identity(rgb, mass_weights, alpha_mass)
                bands_used = 1
                for bands in (4, 16):
                    if mass_error <= 2e-05:
                        break
                    alpha_mass = torch.zeros_like(captured["color"], dtype=torch.float64)
                    for band in range(bands):
                        band_weights = torch.zeros_like(mass_weights)
                        low = band * rgb.shape[1] // bands
                        high = (band + 1) * rgb.shape[1] // bands
                        band_weights[:, low:high, :] = mass_weights[:, low:high, :]
                        alpha_mass += (
                            torch.autograd.grad(
                                rgb, captured["color"], grad_outputs=band_weights, retain_graph=True
                            )[0]
                            .detach()
                            .double()
                        )
                    mass_error = mass_identity(rgb, mass_weights, alpha_mass)
                    bands_used = bands
                if (
                    mass_error > 2e-05
                    or not torch.isfinite(alpha_mass).all()
                    or alpha_mass.min() < -1e-08
                ):
                    raise ValueError("Invalid measured compositing mass; no relaxed check")
                contribution = alpha_mass.mean(1).clamp_min(0).float()
                anchor_mass.index_add_(0, ids // k, contribution)
                slot_mass.index_add_(0, ids, contribution)
                metric_rgb = rgb.clamp(0, 1)
                metric_mse = (
                    (metric_rgb - c.original_image.to("cuda")).square().mean().clamp_min(1e-12)
                )
                image_leaf = metric_rgb.detach().requires_grad_(True)
                perceptual_loss = perceptual(
                    image_leaf[None], c.original_image.to("cuda")[None], normalize=True
                ).mean()
                perceptual_baseline = float(perceptual_loss.detach())
                image_gradient = torch.autograd.grad(perceptual_loss, image_leaf)[0].detach()
                perceptual_derivative = (
                    torch.autograd.grad(
                        metric_rgb,
                        captured["opacity"],
                        grad_outputs=image_gradient,
                        retain_graph=True,
                    )[0]
                    .detach()
                    .flatten()
                )
                perceptual_slot_linear = (
                    -perceptual_derivative
                    * captured["opacity"].detach().flatten()
                    / max(perceptual_baseline, 1e-12)
                )
                perceptual_anchor_linear = torch.zeros(n, device="cuda").index_add_(
                    0, ids // k, perceptual_slot_linear
                )
                del image_leaf, perceptual_loss, image_gradient, perceptual_derivative
                derivative = (
                    torch.autograd.grad(metric_mse, captured["opacity"], retain_graph=True)[0]
                    .detach()
                    .flatten()
                )
                opacity = captured["opacity"].detach().flatten()
                slot_linear = -derivative * opacity
                anchor_linear = torch.zeros(n, device="cuda").index_add_(0, ids // k, slot_linear)
                slot_q, anchor_q, analytic_curvature_audit = analytic_dense_moments(
                    analytic_modules, rgb, ids, n, k
                )
                slot_cost = (slot_linear + slot_q) / metric_mse.detach()
                anchor_cost = (anchor_linear + anchor_q) / metric_mse.detach()
                slot_cost = torch.maximum(slot_cost, perceptual_slot_linear)
                anchor_cost = torch.maximum(anchor_cost, perceptual_anchor_linear)
                if not torch.isfinite(slot_cost).all() or not torch.isfinite(anchor_cost).all():
                    raise ValueError("Nonfinite measured removal response")
                slot_damage.scatter_reduce_(0, ids, slot_cost, reduce="amax", include_self=True)
                anchor_damage[visible] = torch.maximum(anchor_damage[visible], anchor_cost[visible])
                records.append(
                    dict(
                        image=c.image_name,
                        identity_error=identity,
                        mass_identity_error=mass_error,
                        mass_gradient_bands=bands_used,
                        baseline_mse=float(metric_mse),
                        **analytic_curvature_audit,
                        baseline_lpips_normalized=perceptual_baseline,
                        perceptual_removal_risk_first_order=True,
                        predicted_beneficial_rendered_slots=int((slot_cost < 0).sum()),
                        predicted_beneficial_visible_anchors=int((anchor_cost[visible] < 0).sum())
                    )
                )
                captured.clear()
                del pkg, rgb, error, weights, jac, ids, mass_weights, alpha_mass, contribution
                del metric_rgb, metric_mse, derivative, opacity, slot_linear, anchor_linear, slot_q
                del anchor_q, slot_cost, anchor_cost, analytic_curvature_audit
                del perceptual_slot_linear, perceptual_anchor_linear
                from utils.cuda_cache_audit import release_cuda_cache

                release_cuda_cache(
                    output.parent / "cuda_cache_audit.jsonl",
                    "early_scoring:" + context + ":" + c.image_name,
                )
        finally:
            renderer.generate_neural_gaussians = original
        return dict(
            anchor_error=anchor_error,
            frustum=frustum,
            anchor_mass=anchor_mass,
            slot_mass=slot_mass,
            slot_error=slot_error,
            observed=observed,
            anchor_damage=anchor_damage,
            slot_damage=slot_damage,
            records=records,
            wall_seconds=time.time() - capture_start,
        )

    dense_response = measure_response("dense_before_anchor")
    anchor_error = dense_response["anchor_error"]
    frustum = dense_response["frustum"]
    anchor_mass = dense_response["anchor_mass"]
    slot_mass = dense_response["slot_mass"]
    slot_error = dense_response["slot_error"]
    observed = dense_response["observed"]
    anchor_damage = dense_response["anchor_damage"]
    slot_damage = dense_response["slot_damage"]
    records = dense_response["records"]
    dense_observed = observed
    dense_slot_mass = slot_mass
    dense_slot_damage = slot_damage
    slot_response_records = []
    slot_response_audit = None

    @torch.no_grad()
    def evaluate(cams):
        rows = []
        for c in cams:
            visible = renderer.prefilter_voxel(c, model, pipe, bg)
            pkg = renderer.render(c, model, pipe, bg, visible_mask=visible)
            rgb = pkg["render"].clamp(0, 1)
            gt = c.original_image.to("cuda")
            rows.append(
                dict(
                    image=c.image_name,
                    psnr=float(-10 * (rgb - gt).square().mean().clamp_min(1e-12).log10()),
                    ssim=float(ssim(rgb, gt)),
                    l1=float(l1_loss(rgb, gt)),
                    active=int((pkg["radii"] > 0).sum()),
                )
            )
            del pkg, rgb, gt, visible
            from utils.cuda_cache_audit import release_cuda_cache

            release_cuda_cache(
                output.parent / "cuda_cache_audit.jsonl", "early_finite:" + c.image_name
            )
        return dict(
            per_view=rows,
            mean={
                key: float(np.mean([r[key] for r in rows]))
                for key in ("psnr", "ssim", "l1", "active")
            },
        )

    initial = torch.ones(n, k, device="cuda", dtype=torch.bool)
    stages = []
    shared_response_price = None
    from method.shared_price import anchor_price, fixed_price_knee

    try:
        for kind in ("anchor", "slot"):
            select_count = (
                pareto_knee
                if kind == "anchor"
                else lambda rows: fixed_price_knee(rows, shared_response_price)
            )
            base = initial.clone()
            if kind == "anchor":
                eligible = (frustum >= 2).nonzero().flatten()
                priority = anchor_damage
            else:
                if model.slot_keep is None or not torch.equal(model.slot_keep, initial):
                    raise ValueError("Conditional slot measurement requires accepted anchor mask")
                from method.conditional_risk import capture as analytic_slot_capture

                conditional = analytic_slot_capture(model, cameras, pipe, bg, perceptual, output)
                conditional_slot_damage = conditional["slot_damage"]
                slot_response_records = conditional["records"]
                live = initial.flatten()
                measured = (
                    live
                    & torch.isfinite(dense_slot_damage)
                    & torch.isfinite(conditional_slot_damage)
                )
                slot_response_audit = dict(
                    wall_seconds=conditional["wall_seconds"],
                    training_views=len(slot_response_records),
                    surviving_anchors=int(initial.any(1).sum()),
                    changed_risk_rows=int(
                        torch.count_nonzero(
                            conditional_slot_damage[measured] != dense_slot_damage[measured]
                        )
                    ),
                )
                del conditional, live, measured
                priority = slot_damage
                protected = torch.zeros(n * k, device="cuda", dtype=torch.bool)
                strength = priority.reshape(n, k).masked_fill(
                    observed.reshape(n, k) < 2, -torch.inf
                )
                protected[torch.arange(n, device="cuda") * k + strength.argmax(1)] = True
                eligible = ((observed >= 2) & base.flatten() & ~protected).nonzero().flatten()
            local_ids, localization = candidate_indices(
                priority[eligible].clamp_min(0).detach().cpu().numpy()
            )
            eligible = eligible[torch.as_tensor(local_ids, device="cuda", dtype=torch.long)]
            if kind == "slot":
                priority = conditional_slot_damage
                localization["response_context"] = (
                    "fixed_dense_reference_cohort_current_conditional_render_risk"
                )
            if kind == "anchor" and len(eligible) == n:
                eligible = eligible[:-1]
                localization["candidate_rows"] = len(eligible)
                localization["structural_nonempty_anchor_preserved"] = True
            with torch.no_grad():
                if kind == "anchor":
                    centers = model.get_anchor[eligible].detach().cpu().numpy()
                    mass = (anchor_mass / frustum.clamp_min(1))[eligible].cpu().numpy()
                    support = ((frustum > 0) & base.any(1)).nonzero().flatten()
                    support_centers = model.get_anchor[support].detach().cpu().numpy()
                    support_mass = (anchor_mass / frustum.clamp_min(1))[support].cpu().numpy()
                else:
                    parent = eligible // k
                    offset = eligible % k
                    centers = (
                        (
                            model.get_anchor[parent]
                            + model._offset[parent, offset] * model.get_scaling[parent, :3]
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    mass = (slot_mass / observed.clamp_min(1))[eligible].cpu().numpy()
                    support = ((observed > 0) & base.flatten()).nonzero().flatten()
                    sp = support // k
                    so = support % k
                    support_centers = (
                        (model.get_anchor[sp] + model._offset[sp, so] * model.get_scaling[sp, :3])
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    support_mass = (slot_mass / observed.clamp_min(1))[support].cpu().numpy()
            nearest, reliability = surface.correspond(centers)
            support_nearest, support_reliability = surface.correspond(support_centers)
            from method.geometry_risk import rank_anchors as rank
            from method.risk_discount import discounted_risk

            objective_risk, discount_audit = discounted_risk(
                priority[eligible].cpu().numpy(),
                nearest,
                reliability,
                support_nearest,
                support_mass,
            )
            if kind == "slot":
                from method.geometry_risk import rank_slots as rank_slots

                ranking, geometry_audit, static_witness = rank_slots(
                    objective_risk,
                    nearest,
                    mass,
                    reliability,
                    support_nearest,
                    support_mass,
                    support_reliability,
                    teacher_ply=geometry_teacher,
                )
            else:
                ranking, geometry_audit, static_witness = rank(
                    objective_risk,
                    nearest,
                    mass,
                    reliability,
                    support_nearest,
                    support_mass,
                    support_reliability,
                    teacher_ply=geometry_teacher,
                )
            geometry_audit["render_risk_discount"] = discount_audit
            witness = output.with_suffix("." + kind + ".static_geometry.npz")
            np.savez_compressed(
                witness,
                candidate_ids=eligible.cpu().numpy(),
                risk=objective_risk,
                raw_risk=priority[eligible].cpu().numpy(),
                candidate_reliability=reliability,
                group=nearest,
                mass=mass,
                support_group=support_nearest,
                support_mass=support_mass,
                support_confidence=support_reliability,
                rank_order=ranking,
                **static_witness
            )
            from method.io import digest

            geometry_audit.update(witness=str(witness), witness_sha256=digest(witness))
            geometry_audit.update(
                ranking_policy="Static initial geometry penalty plus rendering risk; fixed per-stage scores",
                geometry_returns_direct_candidate_order=True,
                rows=len(eligible),
                support_rows=len(support_mass),
                positive_mass_support_rows=int((support_mass > 0).sum()),
                quantity_target=None,
                support_scope="All current surviving observed support, including noncandidate protected rows",
                changed_order_rows=int(
                    np.count_nonzero(
                        ranking
                        != np.argsort(
                            priority[eligible].cpu().numpy().astype(np.float32), kind="stable"
                        )
                    )
                ),
            )
            geometry_audit["used_for_actual_ranking"] = True
            geometry_audit["early_geometry_policy"] = "static_geometry"
            geometry_audit["noncandidate_support_rows"] = len(support) - len(eligible)
            geometry_audit["includes_one_observation_protected_support"] = True
            del support, support_centers, support_mass, support_nearest
            order = eligible[torch.as_tensor(ranking, device="cuda", dtype=torch.long)]
            upper = len(order)
            counts = sorted(
                set([0, upper // 256, upper // 64, upper // 16, upper // 4, upper // 2, upper])
            )
            curve = []
            baseline = evaluate(admission)
            for count in counts:
                model.slot_keep = base.clone()
                if kind == "anchor":
                    model.slot_keep[order[:count]] = False
                else:
                    model.slot_keep.view(-1)[order[:count]] = False
                curve.append(dict(count=count, **evaluate(fitting)))
            proposed = curve[select_count([r["mean"] for r in curve])]["count"]
            trials = []
            for count in counts:
                if count == 0:
                    continue
                model.slot_keep = base.clone()
                if kind == "anchor":
                    model.slot_keep[order[:count]] = False
                else:
                    model.slot_keep.view(-1)[order[:count]] = False
                trials.append(dict(count=count, **evaluate(admission)))
            full_admission = [dict(count=0, **baseline)] + trials
            pooled = joint_profile(curve, full_admission)
            from utils.adaptive_curve_refinement import refine_profile

            def evaluate_count(count):
                model.slot_keep = base.clone()
                if kind == "anchor":
                    model.slot_keep[order[:count]] = False
                else:
                    model.slot_keep.view(-1)[order[:count]] = False
                return (evaluate(fitting), evaluate(admission))

            curve, trials, pooled, refinement = refine_profile(
                curve, baseline, trials, evaluate_count, joint_profile, select_count
            )
            proposed = curve[select_count([r["mean"] for r in curve])]["count"]
            full_admission = [dict(count=0, **baseline)] + trials
            chosen = select_count([r["mean"] for r in pooled])
            if kind == "anchor":
                shared_response_price = anchor_price([r["mean"] for r in pooled])
            accepted = pooled[chosen]["count"]
            bounded = [r for r in full_admission if r["count"] <= proposed]
            legacy_count = bounded[select_count([r["mean"] for r in bounded])]["count"]
            initial = base.clone()
            if kind == "anchor":
                initial[order[:accepted]] = False
            else:
                initial.view(-1)[order[:accepted]] = False
            model.slot_keep = initial
            stages.append(
                dict(
                    kind=kind,
                    eligible=len(order),
                    data_driven_upper=upper,
                    curve=curve,
                    proposed=proposed,
                    admission_before=baseline,
                    admission_trials=trials,
                    accepted=accepted,
                    joint_curve=pooled,
                    legacy_bounded_accepted_on_this_conditional_mask=legacy_count,
                    curve_refinement=refinement,
                    shared_response_price=shared_response_price,
                    response_price_source="anchor_full_measured_joint_curve",
                    geometry=geometry_audit,
                    render_localization=localization,
                    selection="Anchor full-range knee calibrates one response price; conditional slot coarse and refined choices reuse that price",
                )
            )
        from method.ispg import apply

        initial, inactive_audit = apply(
            model,
            cameras,
            pipe,
            bg,
            initial,
            dense_observed,
            dense_slot_mass,
            geometry_teacher,
            output,
        )
    finally:
        model.slot_keep = None
    if any((not torch.equal(getattr(model, "_" + key), value) for key, value in snapshots.items())):
        raise ValueError("Calibration modified parameters")
    if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(
        gpu_rng, torch.cuda.get_rng_state()
    ):
        raise ValueError("Calibration changed global RNG")
    keep = initial.any(1)
    if not keep.any():
        raise ValueError("No surviving anchor")
    retain_optimizer_rows(model, keep)
    model.slot_keep = initial[keep].contiguous()
    if not model.slot_keep.any(1).all():
        raise ValueError("Lost retained anchor support")
    if any(
        (
            not torch.equal(getattr(model, "_" + key).detach(), value[keep])
            for key, value in snapshots.items()
        )
    ):
        raise ValueError("Survivor parameter rows changed")
    mask_count = int((~model.slot_keep).sum())
    if mask_count != stages[1]["accepted"] + inactive_audit["proposed_retired_slots"]:
        raise ValueError("Slot count not aligned with retained rows")
    report = dict(
        iteration=iteration,
        anchors_before=n,
        anchors_after=len(model.get_anchor),
        removed=int((~keep).sum()),
        slot_removed=mask_count,
        mask_shape=list(model.slot_keep.shape),
        stages=stages,
        training_views=records,
        conditional_slot_training_views=slot_response_records,
        conditional_slot_response=slot_response_audit,
        inactive_geometry=inactive_audit,
        geometry_weight=1.0,
        teacher=teacher_provenance,
        wall_seconds=time.time() - start,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    requirement = dict(
        kind="static_slot_mask",
        version=1,
        storage="PLY slot_bits_* packed uint8",
        n_offsets=k,
        required_renderer_attribute="SLOT_BUDGET_VERSION=1",
    )
    (output.parent / "deployment_requirement.json").write_text(json.dumps(requirement, indent=2))
    output.write_text(json.dumps(report, indent=2))
    from utils.cuda_cache_audit import release_cuda_cache

    release_cuda_cache(output.parent / "cuda_cache_audit.jsonl", "early_complete")
    print("HPRG:", report["removed"], "anchors", mask_count, "slots", flush=True)
    return report
