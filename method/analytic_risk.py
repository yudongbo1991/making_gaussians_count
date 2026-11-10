"""Analytic risk for the final analytic method."""


def moments(modules, rgb, physical_slot_ids, anchors, offsets):
    import torch

    assert offsets == 10 and torch.is_grad_enabled()
    ids = physical_slot_ids
    assert ids.dtype == torch.int64 and ids.ndim == 1 and (ids.numel() > 0)
    assert int(ids.min()) >= 0 and int(ids.max()) < anchors * offsets
    parents = (ids // offsets).contiguous()
    ctx = rgb.grad_fn
    saved = ctx.saved_tensors
    assert type(ctx).__name__ == "_RasterizeGaussiansBackward" and len(saved) == 10
    assert len(saved[0]) == len(ids)
    selected = torch.ones(rgb.shape[1:], device=rgb.device, dtype=torch.int32)
    q_anchor, audit, stats, invalid = modules[0].analyze(
        saved[0].contiguous(),
        saved[7],
        saved[8],
        saved[9],
        rgb.contiguous(),
        ctx.raster_settings.bg.contiguous(),
        parents,
        selected,
        ctx.num_rendered,
        anchors,
        64,
    )
    zero = torch.zeros_like(rgb)
    slot, slot_audit, slot_invalid = modules[1].analyze(
        saved[0].contiguous(),
        saved[7],
        saved[8],
        saved[9],
        rgb.contiguous(),
        zero,
        zero,
        ctx.raster_settings.bg.contiguous(),
        ctx.num_rendered,
    )
    assert int(invalid.sum()) == int(slot_invalid.sum()) == 0
    assert torch.isfinite(q_anchor).all() and torch.isfinite(slot).all()
    assert (q_anchor >= 0).all() and (slot[:, 1] >= 0).all()
    assert float(audit[0].max()) <= 2e-05 and float(audit[1].max()) <= 0.0002
    assert float(slot_audit[0].max()) <= 2e-05 and float(slot_audit[1].max()) <= 0.0002
    diagonal = torch.zeros_like(q_anchor)
    diagonal.index_add_(0, parents, slot[:, 1])
    assert bool((q_anchor <= offsets * diagonal + 1e-12).all())
    record = dict(
        dense_anchor_slot_analytic_curvature=True,
        curvature_samples=0,
        anchor_cross_slot_terms_retained=True,
        workspace_capacity=64,
        analytic_reverse_RGB_max_error=float(audit[0].max()),
        analytic_reverse_T_max_error=float(audit[1].max()),
        overflow_pixels=int((stats[1] > 64).sum()),
        maximum_ray_traversals=int(stats[2].max()),
        analytic_grouped_energy=float(q_anchor.sum()),
        slot_diagonal_energy=float(diagonal.sum()),
    )
    return (slot[:, 1].float(), q_anchor.float(), record)
