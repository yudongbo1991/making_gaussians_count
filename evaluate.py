"""Full TEST PNG metrics, five-pass actual active counts, and saved slot counts."""

from argparse import ArgumentParser, Namespace
import ast
import json
from pathlib import Path
import statistics
import time


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    args = parser.parse_args()
    node = ast.parse((args.model / "cfg_args").read_text(), mode="eval").body
    cfg = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    cfg.update(model_path=str(args.model.resolve()))
    import numpy as np
    import torch
    import lpips
    from plyfile import PlyData
    from arguments import PipelineParams
    from gaussian_renderer import render, prefilter_voxel
    from scene import Scene, GaussianModel
    from utils.image_utils import psnr
    from utils.loss_utils import ssim

    model = GaussianModel(
        cfg["feat_dim"],
        cfg["n_offsets"],
        cfg["voxel_size"],
        cfg["update_depth"],
        cfg["update_init_factor"],
        cfg["update_hierachy_factor"],
        False,
        0,
        cfg["ratio"],
        cfg["add_opacity_dist"],
        cfg["add_cov_dist"],
        cfg["add_color_dist"],
    )
    scene = Scene(Namespace(**cfg), model, load_iteration=args.iteration, shuffle=False)
    model.eval()
    pipeline = PipelineParams(ArgumentParser())
    bg = torch.zeros(3, device="cuda")
    metric = lpips.LPIPS(net="vgg").cuda().eval()
    counts, rows = ({}, [])
    render_times = []
    with torch.no_grad():
        for index, camera in enumerate(scene.getTestCameras()):
            values = []
            for _ in range(5):
                torch.cuda.synchronize()
                started = time.perf_counter()
                visible = prefilter_voxel(camera, model, pipeline, bg)
                pkg = render(camera, model, pipeline, bg, visible_mask=visible)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if _ == 0 and index >= 3:
                    render_times.append(elapsed)
                values.append(int((pkg["radii"] > 0).sum()))
            assert len(set(values)) == 1
            counts[camera.image_name] = values
            image = (pkg["render"].clamp(0, 1) * 255 + 0.5).clamp(0, 255).to(torch.uint8).float()[
                None
            ] / 255
            gt = (camera.original_image.to("cuda").clamp(0, 1) * 255 + 0.5).clamp(0, 255).to(
                torch.uint8
            ).float()[None] / 255
            rows.append(
                dict(
                    image=camera.image_name,
                    psnr=float(psnr(image, gt).mean()),
                    ssim=float(ssim(image, gt)),
                    lpips=float(metric(image, gt)),
                    active=values[0],
                )
            )
    cloud = args.model / f"point_cloud/iteration_{args.iteration}/point_cloud.ply"
    v = PlyData.read(cloud, mmap="r")["vertex"].data
    fields = sorted(
        (n for n in v.dtype.names if n.startswith("slot_bits_")),
        key=lambda n: int(n.rsplit("_", 1)[1]),
    )
    packed = np.stack([v[f] for f in fields], axis=1).astype(np.uint8)
    bits = np.unpackbits(packed, axis=1)
    assert not bits[:, cfg["n_offsets"] :].any()
    summary = dict(
        iteration=args.iteration,
        TEST_views=len(rows),
        metrics={k: statistics.mean((r[k] for r in rows)) for k in ["psnr", "ssim", "lpips"]},
        TEST_mean_active=statistics.mean((r["active"] for r in rows)),
        physical_anchors=len(v),
        enabled_K=int(bits[:, : cfg["n_offsets"]].sum()),
        count_passes=5,
        metric_protocol="uint8 PNG-equivalent RGB; inherited LPIPS(normalize=False); all TEST views",
        per_view=rows,
        five_pass_counts=counts,
    )
    files = [cloud, *sorted(cloud.parent.glob("*.pt"))]
    size_bytes = sum(p.stat().st_size for p in files)
    summary.update(
        saved_representation_bytes=size_bytes,
        saved_representation_MiB=size_bytes / 1024**2,
        representation_files=[p.name for p in files],
        fps=1 / statistics.mean(render_times) if render_times else None,
        fps_views=len(render_times),
        fps_warmup_views=3,
    )
    (args.model / "evaluation.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ["per_view", "five_pass_counts"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
