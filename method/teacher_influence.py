"""Teacher influence for the final analytic method."""

import torch


def identity_error(rgb, color, weights, grad, double):
    if double:
        lhs = (color.detach().double() * grad.double()).sum(0)
        rhs = (rgb.detach().double() * weights.double()).sum((1, 2))
    else:
        lhs = (color.detach() * grad).sum(0)
        rhs = (rgb.detach() * weights).sum((1, 2))
    return float((lhs - rhs).abs().max())


def valid(grad, error):
    return error <= 2e-05 and bool(torch.isfinite(grad).all()) and (float(grad.min()) >= -1e-08)


def partitioned_gradient(rgb, color, weights, bands):
    result = torch.zeros_like(color, dtype=torch.float64)
    height = rgb.shape[1]
    for band in range(bands):
        low, high = (band * height // bands, (band + 1) * height // bands)
        if low == high:
            continue
        local = torch.zeros_like(weights)
        local[:, low:high, :] = weights[:, low:high, :]
        result += (
            torch.autograd.grad(rgb, color, grad_outputs=local, retain_graph=True)[0]
            .detach()
            .double()
        )
    return result


def checked_color_influence(rgb, color, weights):
    grad = torch.autograd.grad(rgb, color, grad_outputs=weights, retain_graph=True)[0].detach()
    error32 = identity_error(rgb, color, weights, grad, False)
    if valid(grad, error32):
        return (
            grad,
            dict(
                accepted_identity_error=error32,
                original_identity_error_fp32=error32,
                gradient_spatial_bands=1,
                identity_reduction="original_float32",
                original_gradient_returned=True,
                identity_tolerance=2e-05,
                negativity_tolerance=-1e-08,
            ),
        )
    error64 = identity_error(rgb, color, weights, grad, True)
    if valid(grad, error64):
        return (
            grad,
            dict(
                accepted_identity_error=error64,
                original_identity_error_fp32=error32,
                gradient_spatial_bands=1,
                identity_reduction="float64",
                original_gradient_returned=True,
                identity_tolerance=2e-05,
                negativity_tolerance=-1e-08,
            ),
        )
    attempts = [dict(bands=1, identity_error_fp32=error32, identity_error_fp64=error64)]
    for bands in [4, 16, 64]:
        precise = partitioned_gradient(rgb, color, weights, bands)
        error = identity_error(rgb, color, weights, precise, True)
        attempts.append(dict(bands=bands, identity_error_fp64=error))
        if valid(precise, error):
            return (
                precise,
                dict(
                    accepted_identity_error=error,
                    original_identity_error_fp32=error32,
                    gradient_spatial_bands=bands,
                    identity_reduction="float64",
                    original_gradient_returned=False,
                    identity_tolerance=2e-05,
                    negativity_tolerance=-1e-08,
                    precision_attempts=attempts,
                ),
            )
    raise RuntimeError(
        "Compositing influence still violates original guards after spatial precision recovery: "
        + str(attempts)
    )
