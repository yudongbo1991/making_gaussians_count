# Upstream attribution and terms: LICENSE.md
"""Student optimization, checkpoint rendering, and inherited metric implementation."""

import os
import torch
import torchvision
import json
import time
from os import makedirs
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render
from scene import Scene, GaussianModel
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import Namespace
from arguments import ModelParams, PipelineParams


def training(
    dataset,
    opt,
    pipe,
    dataset_name,
    testing_iterations,
    saving_iterations,
    logger=None,
    ply_path=None,
):
    from utils.training_resources import TrainingResources

    resource_ledger = TrainingResources()
    first_iter = 0
    optimizer_step_calls_here = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        False,
        0,
        dataset.ratio,
        dataset.add_opacity_dist,
        dataset.add_cov_dist,
        dataset.add_color_dist,
    )
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False)
    if opt.iterations != 30000 or opt.update_until != 15000:
        raise ValueError("Fresh30k and ordinary growth ending by15k required")
    if not dataset.adaptive_teacher_ply:
        raise ValueError("A 15k teacher is required for training")
    if not opt.update_until <= 25000 < opt.iterations:
        raise ValueError("Adaptive recovery must precede30k completion, after densification")
    if not dataset.adaptive_teacher_ply:
        raise ValueError("Geometric recovery needs an audited teacher")
    gaussians.training_setup(opt)
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
        retain_grad = iteration < opt.update_until and iteration >= 0
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            visible_mask=voxel_visible_mask,
            retain_grad=retain_grad,
        )
        (
            image,
            viewspace_point_tensor,
            visibility_filter,
            offset_selection_mask,
            radii,
            scaling,
            opacity,
        ) = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["selection_mask"],
            render_pkg["radii"],
            render_pkg["scaling"],
            render_pkg["neural_opacity"],
        )
        if iteration > 15000:
            history = getattr(gaussians, "ordinary_activation_history", None)
            if history is None:
                raise RuntimeError(
                    "Fresh post-core activation history required; shared resume unsupported"
                )
            history.observe(
                voxel_visible_mask, offset_selection_mask, viewpoint_cam.image_name, iteration
            )
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_loss = 1.0 - ssim(image, gt_image)
        scaling_reg = scaling.prod(dim=1).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01 * scaling_reg
        loss.backward()
        iter_end.record()
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            training_report(
                tb_writer,
                dataset_name,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                None,
                logger,
            )
            if iteration in saving_iterations:
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            if iteration < opt.update_until and iteration > opt.start_stat:
                gaussians.training_statis(
                    viewspace_point_tensor,
                    opacity,
                    visibility_filter,
                    offset_selection_mask,
                    voxel_visible_mask,
                )
                if iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.growth_iteration = iteration
                    gaussians.adjust_anchor(
                        check_interval=opt.update_interval,
                        success_threshold=opt.success_threshold,
                        grad_threshold=opt.densify_grad_threshold,
                        min_opacity=opt.min_opacity,
                    )
                    from utils.cuda_cache_audit import release_cuda_cache

                    release_cuda_cache(
                        Path(dataset.model_path) / "cuda_cache_audit.jsonl",
                        "growth:" + str(iteration),
                    )
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                if hasattr(gaussians, "offset_gradient_squared_accum"):
                    del gaussians.offset_gradient_squared_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                optimizer_step_calls_here += 1
                gaussians.optimizer.zero_grad(set_to_none=True)
            if iteration == 25000:
                from scene.adaptive_recovery import contract_adaptive

                contract_adaptive(
                    gaussians,
                    scene.getTrainCameras(),
                    pipe,
                    background,
                    "geometry",
                    dataset.adaptive_teacher_ply,
                    iteration,
                    os.path.join(dataset.model_path, "ispg_25000.json"),
                )
                from utils.cuda_cache_audit import release_cuda_cache

                release_cuda_cache(
                    Path(dataset.model_path) / "cuda_cache_audit.jsonl", "late_return"
                )
            if iteration == 29000:
                from method.ispg_refresh import check_and_recover

                check_and_recover(
                    gaussians,
                    scene.getTrainCameras(),
                    pipe,
                    background,
                    "geometry",
                    dataset.adaptive_teacher_ply,
                    iteration,
                    dataset.model_path,
                )
            if iteration == 15000:
                from method.hprg import contract

                contract(
                    gaussians,
                    scene.getTrainCameras(),
                    pipe,
                    background,
                    64,
                    iteration,
                    os.path.join(dataset.model_path, "hprg_15000.json"),
                    geometry_teacher=dataset.adaptive_teacher_ply,
                    source_path=dataset.source_path,
                    resolution=dataset.resolution,
                )
                from utils.cuda_cache_audit import release_cuda_cache

                release_cuda_cache(
                    Path(dataset.model_path) / "cuda_cache_audit.jsonl", "early_return"
                )
                from method.activation_history import ActivationHistory

                gaussians.ordinary_activation_history = ActivationHistory(
                    gaussians.slot_keep,
                    [camera.image_name for camera in scene.getTrainCameras()],
                    start_iteration=iteration,
                )
    resource_ledger.finish(dataset.model_path, opt.iterations, len(gaussians.get_anchor))
    resource_path = Path(dataset.model_path, "training_resources.json")
    resource_record = json.loads(resource_path.read_text())
    resource_record.update(
        training_iterations=opt.iterations, optimizer_steps=optimizer_step_calls_here
    )
    resource_path.write_text(json.dumps(resource_record, indent=2))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    tb_writer = None
    print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    dataset_name,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    wandb=None,
    logger=None,
):
    if tb_writer:
        tb_writer.add_scalar(f"{dataset_name}/train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar(
            f"{dataset_name}/train_loss_patches/total_loss", loss.item(), iteration
        )
        tb_writer.add_scalar(f"{dataset_name}/iter_time", elapsed, iteration)
    if iteration in testing_iterations:
        scene.gaussians.eval()
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
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(
                        renderFunc(
                            viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask
                        )["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and idx < 30:
                        tb_writer.add_images(
                            f"{dataset_name}/"
                            + config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        tb_writer.add_images(
                            f"{dataset_name}/"
                            + config["name"]
                            + "_view_{}/errormap".format(viewpoint.image_name),
                            (gt_image[None] - image[None]).abs(),
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                f"{dataset_name}/"
                                + config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                logger.info(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        f"{dataset_name}/" + config["name"] + "/loss_viewpoint - l1_loss",
                        l1_test,
                        iteration,
                    )
                    tb_writer.add_scalar(
                        f"{dataset_name}/" + config["name"] + "/loss_viewpoint - psnr",
                        psnr_test,
                        iteration,
                    )
        if tb_writer:
            tb_writer.add_scalar(
                f"{dataset_name}/" + "total_points", scene.gaussians.get_anchor.shape[0], iteration
            )
        torch.cuda.empty_cache()
        scene.gaussians.train()


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize()
        t_start = time.time()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize()
        t_end = time.time()
        t_list.append(t_end - t_start)
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)
        gt = view.original_image[0:3, :, :].to(rendering.device)
        errormap = (rendering - gt).abs()
        name_list.append("{0:05d}".format(idx) + ".png")
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            errormap, os.path.join(error_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png"))
        per_view_dict["{0:05d}".format(idx) + ".png"] = visible_count.item()
    with open(
        os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), "w"
    ) as fp:
        json.dump(per_view_dict, fp, indent=True)
    return (t_list, visible_count_list)


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train=True,
    skip_test=False,
    wandb=None,
    tb_writer=None,
    dataset_name=None,
    logger=None,
):
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.feat_dim,
            dataset.n_offsets,
            dataset.voxel_size,
            dataset.update_depth,
            dataset.update_init_factor,
            dataset.update_hierachy_factor,
            False,
            0,
            dataset.ratio,
            dataset.add_opacity_dist,
            dataset.add_cov_dist,
            dataset.add_color_dist,
        )
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)
        if not skip_train:
            t_train_list, visible_count = render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
            )
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f"Train FPS: \x1b[1;35m{train_fps.item():.5f}\x1b[0m")
        if not skip_test:
            t_test_list, visible_count = render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
            )
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f"Test FPS: \x1b[1;35m{test_fps.item():.5f}\x1b[0m")
            if tb_writer:
                tb_writer.add_scalar(f"{dataset_name}/test_FPS", test_fps.item(), 0)
    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return (renders, gts, image_names)


def evaluate(
    model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None
):
    full_dict = {}
    per_view_dict = {}
    print("")
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    test_dir = Path(scene_dir) / "test"
    for method in os.listdir(test_dir):
        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        method_dir = test_dir / method
        gt_dir = method_dir / "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)
        ssims = []
        psnrs = []
        lpipss = []
        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
        logger.info(f"model_paths: \x1b[1;35m{model_paths}\x1b[0m")
        logger.info("  SSIM : \x1b[1;35m{:>12.7f}\x1b[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \x1b[1;35m{:>12.7f}\x1b[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \x1b[1;35m{:>12.7f}\x1b[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")
        if tb_writer:
            tb_writer.add_scalar(f"{dataset_name}/SSIM", torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f"{dataset_name}/PSNR", torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f"{dataset_name}/LPIPS", torch.tensor(lpipss).mean().item(), 0)
            tb_writer.add_scalar(
                f"{dataset_name}/VISIBLE_NUMS", torch.tensor(visible_count).mean().item(), 0
            )
        full_dict[scene_dir][method].update(
            {
                "SSIM": torch.tensor(ssims).mean().item(),
                "PSNR": torch.tensor(psnrs).mean().item(),
                "LPIPS": torch.tensor(lpipss).mean().item(),
            }
        )
        per_view_dict[scene_dir][method].update(
            {
                "SSIM": {
                    name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)
                },
                "PSNR": {
                    name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)
                },
                "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                "VISIBLE_COUNT": {
                    name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)
                },
            }
        )
    with open(scene_dir + "/results.json", "w") as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", "w") as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)


def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO)
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)
    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)
    return logger


torch.set_num_threads(4)
TENSORBOARD_FOUND = False
lpips_fn = None
