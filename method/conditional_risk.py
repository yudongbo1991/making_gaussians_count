"""Conditional risk for the final analytic method."""

import math
from pathlib import Path
import time


def raw_risk(values, mse, perceptual_baseline):
    """Analytic signed linear+diagonal RGB risk, versus LP first order.

    The deterministic diagonal RGB term is combined with the normalized
    first-order perceptual term using their maximum.
    """
    import torch

    assert values.ndim == 2 and values.shape[1] == 8 and torch.isfinite(values).all()
    assert (values[:, 1] >= 0).all()
    mse = float(mse)
    perceptual_baseline = float(perceptual_baseline)
    assert (
        math.isfinite(mse)
        and math.isfinite(perceptual_baseline)
        and (mse >= 0)
        and (perceptual_baseline >= 0)
    )
    rgb = (values[:, 0] + values[:, 1]) / max(mse, 1e-12)
    lp = values[:, 2] / max(perceptual_baseline, 1e-12)
    result = torch.maximum(rgb, lp).float()
    assert torch.isfinite(result).all()
    return result


def capture(model, cameras, pipe, bg, perceptual, output):
    from method.native import load_analyzers
    import torch

    module = load_analyzers()[1]
    import gaussian_renderer as renderer
    from utils.cuda_cache_audit import release_cuda_cache

    assert model.slot_keep is not None and (not torch.count_nonzero(bg))
    assert torch.is_grad_enabled(), "Explicit gradient-enabled contract required"
    assert len(cameras) >= 4 and len({c.image_name for c in cameras}) == len(cameras)
    assert cameras == sorted(cameras, key=lambda c: c.image_name)
    n = len(model.get_anchor)
    k = model.n_offsets
    start = time.perf_counter()
    damage = torch.full((n * k,), -torch.inf, device="cuda")
    observed = torch.zeros(n * k, device="cuda", dtype=torch.int32)
    original = renderer.generate_neural_gaussians
    captured = {}
    records = []

    def generate(camera, pc, visible_mask=None, is_training=False):
        with torch.no_grad():
            values = list(original(camera, pc, visible_mask, is_training=True))
        parents = visible_mask.nonzero().flatten()
        captured["ids"] = (parents[:, None] * k + torch.arange(k, device="cuda")).flatten()[
            values[-1]
        ]
        values[1] = values[1].detach().requires_grad_(True)
        values[2] = values[2].detach().requires_grad_(True)
        return tuple(values) if is_training else tuple(values[:5])

    renderer.generate_neural_gaussians = generate
    try:
        for camera in cameras:
            with torch.no_grad():
                visible = renderer.prefilter_voxel(camera, model, pipe, bg)
            pkg = renderer.render(camera, model, pipe, bg, visible_mask=visible)
            rgb = pkg["render"]
            gt = camera.original_image.to("cuda")
            ids = captured["ids"]
            metric = rgb.clamp(0, 1)
            mse = (metric.detach() - gt).square().mean().clamp_min(1e-12)
            if not ids.numel():
                assert not torch.count_nonzero(rgb) and (not pkg["radii"].numel())
                with torch.no_grad():
                    lp = perceptual(metric[None], gt[None], normalize=True).mean()
                assert torch.isfinite(mse) and torch.isfinite(lp)
                records.append(
                    dict(
                        image=camera.image_name,
                        baseline_mse=float(mse),
                        baseline_lpips_normalized=float(lp),
                        curvature_samples=0,
                        analytic_slot_diagonal=True,
                        empty_masked_training_view=True,
                        no_emitted_slot_derivatives=True,
                    )
                )
                del pkg, rgb, gt, ids, metric, mse, lp, visible
                captured.clear()
                release_cuda_cache(
                    Path(output).parent / "cuda_cache_audit.jsonl",
                    "conditional_slot_analytic:empty:" + camera.image_name,
                )
                continue
            leaf = metric.detach().requires_grad_(True)
            lp = perceptual(leaf[None], gt[None], normalize=True).mean()
            gradient = torch.autograd.grad(lp, leaf)[0].detach().contiguous()
            lp_value = float(lp.detach())
            ctx = rgb.grad_fn
            assert type(ctx).__name__ == "_RasterizeGaussiansBackward"
            saved = ctx.saved_tensors
            assert len(saved) == 10
            values, audit, invalid = module.analyze(
                saved[0].contiguous(),
                saved[7],
                saved[8],
                saved[9],
                rgb.contiguous(),
                gt.contiguous(),
                gradient,
                ctx.raster_settings.bg.contiguous(),
                ctx.num_rendered,
            )
            assert (
                int(invalid.sum()) == 0
                and torch.isfinite(values).all()
                and torch.isfinite(audit).all()
            )
            assert values.shape == (len(ids), 8) and (not values.requires_grad)
            assert float(audit[0].max()) <= 2e-05 and float(audit[1].max()) <= 0.0002
            assert (values[:, [1, 3, 5, 6, 7]] >= 0).all()
            cost = raw_risk(values, float(mse), lp_value)
            damage.scatter_reduce_(0, ids, cost, reduce="amax", include_self=True)
            observed.index_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))
            records.append(
                dict(
                    image=camera.image_name,
                    baseline_mse=float(mse),
                    baseline_lpips_normalized=lp_value,
                    curvature_samples=0,
                    analytic_slot_diagonal=True,
                    reverse_RGB_max_error=float(audit[0].max()),
                    reverse_T_max_error=float(audit[1].max()),
                    emitted_slots=len(ids),
                    predicted_beneficial_rendered_slots=int((cost < 0).sum()),
                    no_anchor_derivatives_or_random_projection_backward_calls=True,
                    perceptual_removal_risk_first_order=True,
                )
            )
            del (
                pkg,
                rgb,
                gt,
                ids,
                metric,
                mse,
                leaf,
                lp,
                gradient,
                ctx,
                saved,
                values,
                audit,
                invalid,
                cost,
                visible,
            )
            captured.clear()
            release_cuda_cache(
                Path(output).parent / "cuda_cache_audit.jsonl",
                "conditional_slot_analytic:" + camera.image_name,
            )
    finally:
        renderer.generate_neural_gaussians = original
    assert torch.equal(torch.isfinite(damage), observed > 0)
    return dict(
        slot_damage=damage,
        observed=observed,
        records=records,
        wall_seconds=time.perf_counter() - start,
        original_generator_hook_restored=renderer.generate_neural_gaussians is original,
        only_conditional_slot_response_remeasured=True,
        random_projection_backward_calls=0,
        stage_capture_complete=True,
    )
