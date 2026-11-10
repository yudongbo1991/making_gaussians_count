"""Prepare exact SfM-preserving, teacher-response-ranked occupancy initialization."""

from argparse import ArgumentParser
import ast
import json
from pathlib import Path


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--initial-iteration", type=int, choices=[15000], default=15000)
    parser.add_argument("--voxel", type=float, required=True)
    parser.add_argument("--sfm-multiplier", type=int, choices=[3], default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.voxel <= 0:
        parser.error("Voxel width must be positive.")
    if args.output.exists():
        raise FileExistsError("Prepared inputs must use a fresh directory.")
    import numpy as np
    import torch
    from plyfile import PlyData
    from scipy.special import expit
    from method.rofi import ranked_pool
    from method.io import quantize, keys, save_prior, digest, xyz

    torch.set_num_threads(4)
    root = args.teacher_model.resolve()
    node = ast.parse((root / "cfg_args").read_text(), mode="eval").body
    cfg = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    assert cfg["eval"] and (not cfg["white_background"])
    score = args.scores.resolve()
    meta = json.loads((score.parent / "summary.json").read_text())
    assert meta["test_views_used"] is False
    assert (
        meta["arguments"]["iteration"] == args.initial_iteration
        and meta["arguments"]["views"] == 64
    )
    assert Path(meta["arguments"]["model"]).resolve() == root
    cloud = root / f"point_cloud/iteration_{args.initial_iteration}/point_cloud.ply"
    vertex = PlyData.read(cloud)["vertex"]
    centers = xyz(vertex)
    opacity = expit(np.asarray(vertex["opacity"], dtype=np.float64))
    measured = torch.load(score, map_location="cpu", weights_only=True)
    assert np.array_equal(centers, measured["xyz"].numpy())
    sfm = quantize(xyz(PlyData.read(root / "input.ply")["vertex"]), args.voxel)
    cells, representative, utility, ranked, stats = ranked_pool(
        sfm,
        centers,
        opacity,
        measured["scores"].numpy().astype(np.float64),
        measured["positive_views"].numpy(),
        args.voxel,
    )
    selected = ranked[: (args.sfm_multiplier - 1) * len(sfm)]
    combined = np.unique(np.concatenate([sfm, cells[selected]]), axis=0)
    assert len(combined) == len(sfm) + len(selected) <= args.sfm_multiplier * len(sfm)
    assert np.isin(keys(sfm), keys(combined)).all()
    args.output.mkdir(parents=True)
    item = save_prior(args.output / "render_response", combined, args.voxel)
    sources = [root / "cfg_args", root / "input.ply", cloud, score, score.parent / "summary.json"]
    status_path = root / "exit_status.json"
    if not status_path.exists():
        status_path = root.parent / "exit_status.json"
    status = json.loads(status_path.read_text())
    assert status["exit_code"] == 0
    manifest = dict(
        arguments=dict(
            teacher=str(root),
            source=cfg["source_path"],
            voxel=args.voxel,
            fusion_cap_multiplier=args.sfm_multiplier,
        ),
        teacher_iteration=args.initial_iteration,
        protocols=dict(
            teacher_wall_seconds=status.get("wall_seconds"),
            teacher_training_total_iterations=status.get("teacher_iterations"),
        ),
        source_sha256={str(p): digest(p) for p in sources},
        outputs={"analytic_occupancy": item},
        initial_anchors=len(combined),
        sfm_cells=len(sfm),
        new_occupied_cells=len(selected),
        test_views_used=False,
        base_selection_record=dict(
            stats,
            all_SfM_cells_retained=True,
            initial_offsets_zero=True,
            initial_teacher_iteration=args.initial_iteration,
            fusion_cap_multiplier=args.sfm_multiplier,
        ),
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    np.savez_compressed(
        args.output / "selection.npz",
        selected_cells=cells[selected],
        teacher_rows=representative[selected],
        utility=utility[selected],
    )
    assert not torch.cuda.is_initialized()
    print(
        json.dumps(
            dict(prior=item["prior"], initial_anchors=len(combined), sfm_cells=len(sfm)), indent=2
        )
    )


if __name__ == "__main__":
    main()
