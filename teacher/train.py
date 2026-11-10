# Upstream 2DGS attribution and terms: LICENSE.md
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams


def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0
        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg["rend_normal"]
        surf_normal = render_pkg["surf_normal"]
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * normal_error.mean()
        dist_loss = lambda_dist * rend_dist.mean()
        total_loss = loss + dist_loss + normal_loss
        total_loss.backward()
        iter_end.record()
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            training_report(
                None,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                reference_camera = scene.getTrainCameras()[0]
                reference_rgb = render(reference_camera, gaussians, pipe, background)[
                    "render"
                ].clone()
                scene.save(iteration)
                reload_model = GaussianModel(dataset.sh_degree)
                reload_model.load_ply(
                    os.path.join(
                        scene.model_path,
                        "point_cloud",
                        "iteration_" + str(iteration),
                        "point_cloud.ply",
                    )
                )
                reloaded_rgb = render(reference_camera, reload_model, pipe, background)["render"]
                reload_error = float((reference_rgb - reloaded_rgb).abs().max())
                if reload_error != 0:
                    raise RuntimeError(
                        "Saved teacher RGB differs from save-stage state: " + str(reload_error)
                    )
                import json

                with open(
                    os.path.join(
                        scene.model_path, "saved_model_validation_" + str(iteration) + ".json"
                    ),
                    "w",
                ) as proof:
                    json.dump(
                        dict(
                            iteration=iteration,
                            image=reference_camera.image_name,
                            reload_RGB_max_error=reload_error,
                            save_stage="before_same_iteration_densification",
                            test_views_used=False,
                        ),
                        proof,
                        indent=2,
                    )
                del reference_rgb, reloaded_rgb, reload_model
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.opacity_cull,
                        scene.cameras_extent,
                        size_threshold,
                    )
                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)


def prepare_output_and_logger(args):
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as stream:
        stream.write(str(Namespace(**vars(args))))
    return None


@torch.no_grad()
def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
):
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
    )
    print("\nTraining complete.")
