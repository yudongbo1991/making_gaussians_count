"""Shared price for the final analytic method."""

import numpy as np


def arrays(rows):
    if not rows:
        raise ValueError("Nonempty measured profile required")
    active = np.asarray([r["active"] for r in rows], dtype=np.float64)
    psnr = np.asarray([r["psnr"] for r in rows], dtype=np.float64)
    if not np.isfinite(active).all() or not np.isfinite(psnr).all() or (active < 0).any():
        raise ValueError("Finite nonnegative counts and finite PSNR required")
    return (active, psnr)


def anchor_price(rows):
    active, psnr = arrays(rows)
    return max(float((psnr[0] - psnr).max()), 1e-12) / max(float((active[0] - active).max()), 1.0)


def fixed_price_knee(rows, price):
    if not np.isfinite(price) or price < 0:
        raise ValueError("Finite nonnegative shared response price required")
    active, psnr = arrays(rows)
    saving = active[0] - active
    damage = np.maximum(psnr[0] - psnr, 0.0)
    benefit = price * saving
    utility = benefit - damage
    roundoff = 64 * np.finfo(float).eps * (np.abs(benefit) + np.abs(damage))
    utility[np.abs(utility) <= roundoff] = 0.0
    useful = saving > 0
    if not useful.any():
        return 0
    utility[~useful] = -np.inf
    best = int(np.argmax(utility))
    return best if utility[best] > 0 else 0


def verify_price_consistency(early):
    assert [s["kind"] for s in early["stages"]] == ["anchor", "slot"]
    a, b = early["stages"]
    price = anchor_price([r["mean"] for r in a["joint_curve"]])
    assert a["shared_response_price"] == b["shared_response_price"] == price
    assert (
        a["response_price_source"]
        == b["response_price_source"]
        == "anchor_full_measured_joint_curve"
    )
    for stage in [a, b]:
        profile = stage["joint_curve"]
        selected = fixed_price_knee([r["mean"] for r in profile], price)
        assert profile[selected]["count"] == stage["accepted"]
    return price
