"""Static geometric removal risk with fixed weight 1.0, recomputed for each hierarchy level."""

import time
import numpy as np
import method.teacher_graph as native
import method.support_objective as quadratic
import method.mass_budget as mass_weighted
from method.io import digest

GEOMETRY_WEIGHT = 1.0


def scores(data):
    n = len(data["risk"])
    geometry = np.empty(n, dtype=np.float64)
    for start in range(0, n, 65536):
        stop = min(start + 65536, n)
        groups = data["group"][start:stop]
        targets = data["target"][groups]
        fractions = data["share"][start:stop, None] * data["group_fraction"][groups]
        geometry[start:stop] = data["risk_scale"] * np.sum(
            data["target_budget"][targets] * fractions * fractions, axis=1
        )
    total = data["risk"] + GEOMETRY_WEIGHT * geometry
    if not np.isfinite(geometry).all() or (geometry < 0).any() or (not np.isfinite(total).all()):
        raise ValueError("Finite fixed scores and nonnegative geometry required")
    return (geometry, total)


def prepare(kind, *args, teacher_ply):
    if kind not in ("anchor", "slot"):
        raise ValueError("Declared anchor or active-slot stage required")
    teacher_hash = digest(teacher_ply)
    base = quadratic.original_prepare(*args)
    surface = native._surface(str(teacher_ply), teacher_hash)
    started = time.perf_counter()
    ids, transition, graph = native.transition(surface, base["group_names"])
    seconds = time.perf_counter() - started
    data = quadratic.prepare(*args, ids, transition)
    if kind == "slot":
        data = mass_weighted.reweight(data)
    assert np.isclose(
        data["risk_scale"] * data["target_budget"].sum(),
        base["risk_scale"] * base["weight"].sum(),
        rtol=2e-12,
        atol=1e-12,
    )
    return (
        data,
        ids,
        transition,
        dict(teacher_sha256=teacher_hash, graph=graph, teacher_graph_wall_seconds=seconds),
    )


def rank(kind, *args, teacher_ply):
    started = time.perf_counter()
    data, ids, transition, graph = prepare(kind, *args, teacher_ply=teacher_ply)
    geometry, total = scores(data)
    order = np.argsort(total, kind="stable")
    assert len(np.unique(order)) == len(total)
    audit = dict(
        **graph,
        active_stage=kind,
        geometry_weight=GEOMETRY_WEIGHT,
        risk_scale=float(data["risk_scale"]),
        geometric_score_sum=float(geometry.sum()),
        total_score_sum=float(total.sum()),
        ranking_seconds=time.perf_counter() - started
    )
    witness = dict(
        static_geometric_cost=geometry,
        static_total_score=total,
        static_geometry_weight=np.asarray(GEOMETRY_WEIGHT),
        teacher_target_ids=ids,
        teacher_transition=transition,
    )
    return (order, audit, witness)


def rank_anchors(*args, teacher_ply):
    return rank("anchor", *args, teacher_ply=teacher_ply)


def rank_slots(*args, teacher_ply):
    return rank("slot", *args, teacher_ply=teacher_ply)
