# Upstream attribution and terms: LICENSE.md
"""Curve selection and teacher provenance for the fixed final method."""

import json
import ast
import hashlib
from pathlib import Path
import numpy as np
from method.ispg_history import contract_adaptive as contract_adaptive

__all__ = ["pareto_knee", "teacher_contract", "contract_adaptive"]


def pareto_knee(rows):
    """Maximum normalized savings minus damage; no fixed deletion fraction."""
    origin = rows[0]
    saving = np.array([origin["active"] - r["active"] for r in rows])
    damage = np.array([origin["psnr"] - r["psnr"] for r in rows])
    useful = saving > 0
    if not useful.any():
        return 0
    benefit = saving / max(float(saving.max()), 1.0)
    risk = np.maximum(damage, 0) / max(float(damage.max()), 1e-12)
    utility = benefit - risk
    utility[~useful] = -np.inf
    best = int(np.argmax(utility))
    return best if utility[best] > 0 else 0


def teacher_contract(teacher_ply, source_path, resolution):
    path = Path(teacher_ply).resolve()
    parent = path.parents[2]
    node = ast.parse((parent / "cfg_args").read_text(), mode="eval").body
    if (
        not isinstance(node, ast.Call)
        or not isinstance(node.func, ast.Name)
        or node.func.id != "Namespace"
        or node.args
    ):
        raise ValueError("Invalid teacher configuration")
    cfg = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    if not cfg["eval"] or cfg["white_background"] or cfg["resolution"] != resolution:
        raise ValueError("Held-out black-background teacher with matching resolution required")
    if Path(cfg["source_path"]).resolve() != Path(source_path).resolve():
        raise ValueError("Teacher and student sources differ")
    status = parent / "exit_status.json"
    if not status.exists():
        status = parent.parent / "exit_status.json"
    if json.loads(status.read_text())["exit_code"] != 0:
        raise ValueError("Completed teacher required")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return dict(
        teacher_ply=str(path),
        sha256=h.hexdigest(),
        teacher_eval=True,
        source_path=str(Path(source_path).resolve()),
        resolution=resolution,
    )
