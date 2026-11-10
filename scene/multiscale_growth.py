"""Uncapped RMS growth with finest eligible witness deduplication and teacher-supported birth offsets."""

import json
import torch
from torch_scatter import scatter_max
from utils.general_utils import inverse_sigmoid


def empty_cells(existing, candidates):
    """Exact integer-row membership without quadratic broadcast buffers."""
    if not len(candidates):
        return torch.zeros(0, device=existing.device, dtype=torch.bool)
    _, ids = torch.unique(torch.cat([existing, candidates]), dim=0, return_inverse=True)
    occupied = torch.zeros(int(ids.max()) + 1, device=ids.device, dtype=torch.bool)
    occupied[ids[: len(existing)]] = True
    return ~occupied[ids[len(existing) :]]


@torch.no_grad()
def grow(model, grads, threshold, offset_mask):
    before = len(model.get_anchor)
    iteration = getattr(model, "growth_iteration", 0)
    if grads.shape != (before * model.n_offsets,) or offset_mask.shape != grads.shape:
        raise ValueError("Statistics must refer to the complete pre-growth model")
    if not torch.isfinite(grads).all():
        raise ValueError("Nonfinite proposal priorities")
    audit = dict(
        strategy="uncapped_response_spatial_dedup",
        iteration=getattr(model, "growth_iteration", None),
        anchors_before=before,
        budget=None,
        room=None,
        levels=[],
        born=0,
    )
    audit["cutoff_policy"] = "hierarchical"
    audit["demand_pooling"] = "RMS"
    audit["demand_statistics"] = model.growth_demand_diagnostics

    def record():
        path = getattr(model, "growth_audit_path", "")
        if path:
            with open(path, "a") as f:
                f.write(json.dumps(audit) + "\n")

    xyz = (model.get_anchor[:, None] + model._offset * model.get_scaling[:, None, :3]).flatten(0, 1)
    requests = []
    for level in range(model.update_depth):
        size_factor = model.update_init_factor // model.update_hierachy_factor**level
        cell_size = model.voxel_size * size_factor
        if cell_size <= 0:
            raise ValueError("Invalid growth grid size")
        cutoff = threshold * (model.update_hierachy_factor // 2) ** level
        chosen = (grads >= cutoff) & offset_mask
        chosen &= torch.rand_like(grads) > 0.5 ** (level + 1)
        sources = chosen.nonzero().flatten()
        if not len(sources):
            audit["levels"].append(
                dict(
                    level=level,
                    cell_size=cell_size,
                    gradient_cutoff=cutoff,
                    selected_offsets=0,
                    candidate_cells=0,
                    empty_cells=0,
                    admitted=0,
                )
            )
            continue
        cells, inverse = torch.unique(
            torch.round(xyz[sources] / cell_size).int(), dim=0, return_inverse=True
        )
        free = empty_cells(torch.round(model.get_anchor / cell_size).int(), cells)
        free_ids = free.nonzero().flatten()
        score, witness = scatter_max(grads[sources], inverse, dim=0)
        audit["levels"].append(
            dict(
                level=level,
                cell_size=cell_size,
                gradient_cutoff=cutoff,
                selected_offsets=len(sources),
                candidate_cells=len(cells),
                empty_cells=len(free_ids),
                admitted=0,
            )
        )
        if len(free_ids):
            requests.append(
                dict(
                    level=level,
                    size=cell_size,
                    cells=cells[free_ids],
                    score=score[free_ids],
                    witness=sources[witness[free_ids]],
                    sources=sources,
                    inverse=inverse,
                    free_ids=free_ids,
                )
            )
    if not requests:
        audit["reason"] = "No empty eligible cell"
        record()
        return
    scores = torch.cat([r["score"] for r in requests])
    witnesses = torch.cat([r["witness"] for r in requests])
    levels = torch.cat([torch.full_like(r["witness"], r["level"]) for r in requests])
    _, source_inverse = torch.unique(witnesses, return_inverse=True)
    row_ids = torch.arange(len(scores), device=grads.device)
    representative = scatter_max(row_ids, source_inverse, dim=0)[0]
    audit["proposals_before_source_dedup"] = len(scores)
    audit["unique_source_requests"] = len(representative)
    ordering = torch.argsort(scores[representative], descending=True, stable=True)
    ranked = representative[ordering]
    proposed_positions = torch.cat([r["cells"].to(xyz.dtype) * r["size"] for r in requests])
    _, position_inverse = torch.unique(proposed_positions[ranked], dim=0, return_inverse=True)
    rank_priority = len(ranked) - torch.arange(len(ranked), device=grads.device)
    first_at_position = scatter_max(rank_priority, position_inverse, dim=0)[1]
    unique_rank = torch.zeros(len(ranked), device=grads.device, dtype=torch.bool)
    unique_rank[first_at_position] = True
    ranked = ranked[unique_rank]
    audit["unique_position_requests"] = len(ranked)
    accepted = ranked
    audit["selected_priority_min"] = float(scores[accepted].min())
    audit["selected_priority_max"] = float(scores[accepted].max())
    audit["selected_source_sample"] = witnesses[accepted[:16]].tolist()
    if len(torch.unique(witnesses[accepted])) != len(accepted):
        raise RuntimeError("One Gaussian consumed multiple proposal slots")
    positions, features, scales, priorities, offsets = ([], [], [], [], [])
    guidance = getattr(model, "birth_surface_guidance", None)
    if guidance is not None:
        audit["geometry_birth_guidance"] = []
    cursor = 0
    mode = "none"
    for request in requests:
        selected = (
            accepted[(accepted >= cursor) & (accepted < cursor + len(request["score"]))] - cursor
        )
        cursor += len(request["score"])
        audit["levels"][request["level"]]["admitted"] = len(selected)
        if not len(selected):
            continue
        parent = request["witness"][selected] // model.n_offsets
        position = request["cells"][selected].to(xyz.dtype) * request["size"]
        source_features = model._anchor_feat[request["sources"] // model.n_offsets]
        pooled = scatter_max(
            source_features, request["inverse"][:, None].expand_as(source_features), dim=0
        )[0]
        feature = pooled[request["free_ids"][selected]]
        positions.append(position)
        features.append(feature)
        scales.append(torch.full((len(selected), 6), request["size"], device=grads.device).log())
        priorities.append(request["score"][selected])
        if guidance is None:
            offsets.append(torch.zeros(len(selected), model.n_offsets, 3, device=grads.device))
        else:
            normalized, evidence = guidance.initialize_normalized(
                position, scales[-1], request["size"], model.n_offsets
            )
            offsets.append(normalized)
            audit["geometry_birth_guidance"].append(evidence)
    position = torch.cat(positions)
    count = len(position)
    rotation = torch.zeros(count, 4, device=grads.device)
    rotation[:, 0] = 1
    tensors = dict(
        anchor=position,
        anchor_feat=torch.cat(features),
        scaling=torch.cat(scales),
        rotation=rotation,
        offset=torch.cat(offsets),
        opacity=inverse_sigmoid(torch.full((count, 1), 0.1, device=grads.device)),
    )
    if not all((torch.isfinite(x).all() for x in tensors.values())):
        raise RuntimeError("Invalid growth transaction")
    updated = model.cat_tensors_to_optimizer(tensors)
    for name in ["anchor", "anchor_feat", "scaling", "rotation", "offset", "opacity"]:
        setattr(model, "_" + name, updated[name])
    model.anchor_demon = torch.cat([model.anchor_demon, grads.new_zeros(count, 1)])
    model.opacity_accum = torch.cat([model.opacity_accum, grads.new_zeros(count, 1)])
    audit.update(born=count, anchors_after=len(model.get_anchor))
    record()
