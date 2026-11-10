# Upstream attribution and terms: LICENSE.md
"""Retain parameter rows and the aligned Adam state after anchor retirement."""

import torch


@torch.no_grad()
def retain_optimizer_rows(model, keep):
    names = ["anchor", "offset", "anchor_feat", "scaling", "rotation", "opacity"]
    n = len(model.get_anchor)
    if keep.shape != (n,) or keep.dtype != torch.bool or (not keep.any()):
        raise ValueError("Invalid anchor retention mask")
    groups = [g for g in model.optimizer.param_groups if g["name"] in names]
    if sorted((g["name"] for g in groups)) != sorted(names):
        raise ValueError("Incomplete anchor optimizer groups")
    for group in groups:
        if len(group["params"]) != 1 or group["params"][0].shape[0] != n:
            raise ValueError("Expected one aligned row tensor per anchor field")
        if group["params"][0].grad is not None:
            raise ValueError("Contraction must follow optimizer step and gradient clearing")
    for group in model.optimizer.param_groups:
        name = group["name"]
        if name not in names:
            continue
        if len(group["params"]) != 1:
            raise ValueError("Expected one row tensor per anchor field")
        old = group["params"][0]
        if old.grad is not None:
            raise ValueError("Contraction must follow optimizer step and gradient clearing")
        new = torch.nn.Parameter(old.detach()[keep].clone(), requires_grad=old.requires_grad)
        state = model.optimizer.state.pop(old, None)
        group["params"][0] = new
        setattr(model, "_" + name, new)
        if state is not None:
            model.optimizer.state[new] = {
                key: (
                    value[keep].clone()
                    if torch.is_tensor(value) and value.shape == old.shape
                    else value.clone() if torch.is_tensor(value) else value
                )
                for key, value in state.items()
            }
    for name in ["anchor_demon", "opacity_accum", "max_radii2D"]:
        value = getattr(model, name, None)
        if value is not None and len(value) == n:
            setattr(model, name, value[keep].clone())
    for name in ["offset_gradient_accum", "offset_denom"]:
        value = getattr(model, name, None)
        if value is not None and len(value) == n * model.n_offsets:
            setattr(
                model,
                name,
                value.reshape(n, model.n_offsets, -1)[keep].reshape(-1, value.shape[-1]).clone(),
            )
