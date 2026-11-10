# Upstream attribution and terms: LICENSE.md
"""TRAIN-only teacher compositing influence for occupancy fusion;64 views by default."""

import argparse
import json
from pathlib import Path
import sys
from method.io import config_namespace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--iteration", type=int, choices=[15000], default=15000)
    p.add_argument("--views", type=int, default=64, help="0 means all training cameras")
    a = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent / "teacher"))
    import numpy as np
    import torch
    from scene import Scene, GaussianModel
    import utils.camera_utils as camera_utils

    original_load = camera_utils.loadCam

    def load_and_close(*args, **kwargs):
        info = args[2] if len(args) > 2 else kwargs["cam_info"]
        camera = original_load(*args, **kwargs)
        info.image.close()
        return camera

    camera_utils.loadCam = load_and_close
    from arguments import PipelineParams
    from gaussian_renderer import render
    from utils.sh_utils import eval_sh

    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.5)
    cfg = config_namespace(Path(a.model) / "cfg_args")
    cfg.update(model_path=a.model, data_device="cpu")
    if not cfg["eval"]:
        raise ValueError("Teacher must have a held-out evaluation split")
    m = GaussianModel(cfg["sh_degree"])
    scene = Scene(argparse.Namespace(**cfg), m, load_iteration=a.iteration, shuffle=False)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    train = scene.getTrainCameras()
    cameras = (
        train
        if a.views == 0
        else [train[i] for i in np.linspace(0, len(train) - 1, min(a.views, len(train)), dtype=int)]
    )
    pipe = PipelineParams(argparse.ArgumentParser())
    pipe.depth_ratio = 0.0
    if cfg["white_background"]:
        raise ValueError("Current identity check assumes black background")
    bg = torch.zeros(3, device="cuda")
    scores = torch.zeros((len(m.get_xyz), 3), device="cuda")
    views = torch.zeros(len(m.get_xyz), device="cuda", dtype=torch.int32)
    per_view = []
    for value in vars(m).values():
        if isinstance(value, torch.Tensor) and value.is_leaf:
            value.requires_grad_(False)
    for i, camera in enumerate(cameras):
        with torch.no_grad():
            sh = m.get_features.transpose(1, 2)
            direction = m.get_xyz - camera.camera_center[None, :]
            direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-08)
            color = (eval_sh(m.active_sh_degree, sh, direction) + 0.5).clamp_min(0.0)
            reference = render(camera, m, pipe, bg)["render"] if i == 0 else None
        color = color.detach().requires_grad_(True)
        result = render(camera, m, pipe, bg, override_color=color)
        rgb = result["render"]
        with torch.no_grad():
            if reference is not None:
                equivalence = float((reference - rgb).abs().max())
                if equivalence > 0.0002:
                    raise RuntimeError(
                        f"RGB override is not equivalent to SH rendering: {equivalence}"
                    )
            gt = camera.original_image.to("cuda")
            error = (rgb.clamp(0, 1) - gt).abs().mean(0)
            dx = torch.nn.functional.pad((gt[:, :, 1:] - gt[:, :, :-1]).abs().mean(0), (0, 1))
            dy = torch.nn.functional.pad((gt[:, 1:, :] - gt[:, :-1, :]).abs().mean(0), (0, 0, 0, 1))
            edge = dx + dy
            edge = edge > torch.quantile(edge.flatten(), 0.75)
            weights = torch.stack([torch.ones_like(error), error, error * edge]) / error.numel()
        from method.teacher_influence import checked_color_influence

        grad, influence_audit = checked_color_influence(rgb, color, weights)
        with torch.no_grad():
            check = influence_audit["accepted_identity_error"]
            scores += grad
            views += (grad[:, 0] > 1e-10).int()
            record = dict(
                image=camera.image_name,
                positive_contributors=int((grad[:, 0] > 1e-10).sum()),
                alpha_mass=float(grad[:, 0].sum()),
                mean_abs_error=float(error.mean()),
                identity_error=check,
            )
            record["numerical_influence"] = influence_audit
            per_view.append(record)
            if i % 10 == 0:
                print(i + 1, len(cameras), record, flush=True)
        del rgb, result, color, grad, gt, weights, reference
    torch.save(
        dict(xyz=m.get_xyz.detach().cpu(), scores=scores.cpu(), positive_views=views.cpu()),
        out / "scores.pt",
    )
    report = dict(
        arguments=vars(a),
        test_views_used=False,
        per_view=per_view,
        teacher_points=len(m.get_xyz),
        points_supported_in_two_views=int((views >= 2).sum()),
        source="Exact color derivative of the official 2DGS compositor; checked against rendered RGB.",
        error_source="teacher",
        caveat="Contribution indicates rendered occupancy, not true geometry or guaranteed future improvement.",
    )
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print("done", report["points_supported_in_two_views"], flush=True)


if __name__ == "__main__":
    main()
