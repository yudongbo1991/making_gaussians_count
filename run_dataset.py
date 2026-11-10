"""Run the paper configuration on one or more scenes of a dataset."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DATASETS = ("mipnerf360", "tanks_and_temples", "deep_blending", "bungeenerf")


def commands(config, scene, data_root, output_root, teacher_root=None):
    source = (data_root / scene["name"]).resolve()
    work = (output_root / config["dataset"] / scene["name"]).resolve()
    teacher = (teacher_root / scene["name"]).resolve() if teacher_root else work / "teacher"
    score, prior, student = work / "response", work / "initialization", work / "student"
    python = sys.executable

    def command(script, *args):
        return [python, "-u", str(ROOT / script), *map(str, args)]

    return {
        "teacher": command(
            "train_teacher.py",
            "-s",
            source,
            "-m",
            teacher,
            "--images",
            scene["images"],
            "--resolution",
            scene["resolution"],
            "--iterations",
            config["teacher_iterations"],
        ),
        "score": command(
            "score_teacher.py",
            "--model",
            teacher,
            "--iteration",
            config["teacher_iterations"],
            "--views",
            64,
            "--output",
            score,
        ),
        "prepare": command(
            "prepare.py",
            "--teacher-model",
            teacher,
            "--scores",
            score / "scores.pt",
            "--initial-iteration",
            config["teacher_iterations"],
            "--voxel",
            config["voxel_size"],
            "--sfm-multiplier",
            config["sfm_capacity_multiplier"],
            "--output",
            prior,
        ),
        "train": command(
            "train.py",
            "-s",
            source,
            "-m",
            student,
            "--images",
            scene["images"],
            "--resolution",
            scene["resolution"],
            "--voxel_size",
            config["voxel_size"],
            "--feat_dim",
            config["feature_dim"],
            "--n_offsets",
            config["slots_per_anchor"],
            "--seed",
            config["seed"],
            "--iterations",
            config["student_iterations"],
            "--surface_prior",
            prior / "render_response/prior.npz",
            "--adaptive_teacher_ply",
            teacher / "point_cloud/iteration_15000/point_cloud.ply",
            "--quiet",
        ),
        "evaluate": command("evaluate.py", "--model", student, "--iteration", 30000),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing the dataset scene folders.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--teacher-root",
        type=Path,
        help="Optional directory of existing 15k teacher models, one folder per scene.",
    )
    parser.add_argument("--scenes", nargs="+", help="Default: every scene in the selected dataset.")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["teacher", "score", "prepare", "train", "evaluate"],
        default=["teacher", "score", "prepare", "train", "evaluate"],
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "configs" / f"{args.dataset}.json").read_text())
    fixed = dict(
        feature_dim=32,
        slots_per_anchor=10,
        teacher_iterations=15000,
        student_iterations=30000,
        sfm_capacity_multiplier=3,
        hprg_iteration=15000,
        ispg_iterations=[15000, 25000, 29000],
        geometry_weight=1.0,
    )
    if any(cfg[key] != value for key, value in fixed.items()):
        parser.error("The configuration differs from the supported final method.")
    if args.scenes and set(args.scenes) - {s["name"] for s in cfg["scenes"]}:
        parser.error("A requested scene is absent from the dataset configuration.")
    if args.teacher_root and "teacher" in args.stages:
        parser.error("With --teacher-root, select --stages score prepare train evaluate.")
    for scene in cfg["scenes"]:
        if args.scenes and scene["name"] not in args.scenes:
            continue
        pipeline = commands(cfg, scene, args.data_root, args.output_root, args.teacher_root)
        if not args.dry_run:
            source = args.data_root / scene["name"]
            if not (source / "sparse/0").is_dir() or not (source / scene["images"]).is_dir():
                raise FileNotFoundError(f"COLMAP scene or image folder missing: {source}")
        for stage in args.stages:
            print(f'[{scene["name"]}: {stage}] {shlex.join(pipeline[stage])}', flush=True)
            if not args.dry_run:
                subprocess.run(
                    pipeline[stage],
                    cwd=ROOT,
                    check=True,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONPATH=str(ROOT)),
                )


if __name__ == "__main__":
    main()
