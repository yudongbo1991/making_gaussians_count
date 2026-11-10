# Upstream attribution and terms: LICENSE.md
"""Validate prepared prior provenance before any GPU initialization."""

import ast
import hashlib
import json
from pathlib import Path
import numpy as np


def validate_prior(args):
    prior = Path(args.surface_prior).resolve()
    meta = json.loads((prior.parent.parent / "manifest.json").read_text())
    variants = [
        (name, item)
        for name, item in meta["outputs"].items()
        if Path(item["prior"]).resolve() == prior
    ]
    if len(variants) != 1:
        raise ValueError("Prior not registered in prepared controls")
    name, item = variants[0]
    actual = hashlib.sha256(prior.read_bytes()).hexdigest()
    if actual != item["sha256"]:
        raise ValueError("Prior hash changed")
    teacher = Path(meta["arguments"]["teacher"])
    cfg_path = teacher / "cfg_args"
    if hashlib.sha256(cfg_path.read_bytes()).hexdigest() != meta["source_sha256"][str(cfg_path)]:
        raise ValueError("Teacher configuration changed")
    node = ast.parse(cfg_path.read_text(), mode="eval").body
    if (
        not isinstance(node, ast.Call)
        or not isinstance(node.func, ast.Name)
        or node.func.id != "Namespace"
        or node.args
    ):
        raise ValueError("Invalid teacher configuration")
    cfg = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    if not args.eval or args.lod != 0 or (not cfg["eval"]):
        raise ValueError("Held-out evaluation required")
    for key in ["source_path", "images", "resolution", "white_background"]:
        if getattr(args, key) != cfg[key]:
            raise ValueError("Teacher/student configuration mismatch: " + key)
    with np.load(prior) as f:
        anchors = f["anchors"]
        width = float(f["voxel"])
        if (
            len(anchors) != item["anchors"]
            or anchors.shape[1:] != (3,)
            or (not np.isfinite(anchors).all())
        ):
            raise ValueError("Invalid prepared geometry")
        if not np.isclose(width, args.voxel_size, rtol=1e-06):
            raise ValueError("Grid mismatch")
        count = len(anchors)
    return dict(
        prior_path=str(prior),
        sha256=actual,
        selection=name,
        initial_anchors=count,
        extra_teacher_steps=int(meta.get("teacher_iteration", 15000)),
        initial_teacher_iteration=int(meta.get("teacher_iteration", 15000)),
        teacher_training_total_iterations=meta["protocols"].get(
            "teacher_training_total_iterations", 15000
        ),
        teacher_wall_seconds=meta["protocols"]["teacher_wall_seconds"],
        inherited_offsets=False,
        inherited_features=False,
        caveat="All SfM cells plus a TRAIN-response-ranked prefix of novel teacher occupied cells; initial slot offsets and student features are zero.",
    )
