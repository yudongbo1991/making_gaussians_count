#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""Fresh30k training for the final analytic method. Set CUDA_VISIBLE_DEVICES externally."""

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import random
from arguments import ModelParams, OptimizationParams, PipelineParams


def main():
    parser = ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    optimizer = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[15000, 25000, 30000])
    args = parser.parse_args()
    if (
        not args.source_path
        or not args.model_path
        or (not args.surface_prior)
        or (not args.adaptive_teacher_ply)
    ):
        parser.error("Provide -s, -m, --surface_prior and --adaptive_teacher_ply.")
    if args.iterations != 30000 or args.update_until != 15000 or args.seed < 0:
        parser.error(
            "The final method uses a fresh30k run, growth ending at15k, and a nonnegative seed."
        )
    if args.white_background or args.n_offsets != 10 or (not args.eval) or (args.lod != 0):
        parser.error(
            "The validated final protocol requires black background,10 slots, eval split, and lod0."
        )
    output = Path(args.model_path).resolve()
    if (output / "cfg_args").exists():
        raise FileExistsError(
            "Use a fresh output directory; final training does not overwrite or resume."
        )
    output.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    import lpips
    import engine
    from utils.general_utils import safe_state

    engine.lpips_fn = lpips.LPIPS(net="vgg").to("cuda")
    safe_state(args.quiet)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    (output / "random_seed.json").write_text(
        json.dumps(
            dict(
                seed=args.seed,
                torch_cpu_rng_sha256=hashlib.sha256(
                    torch.get_rng_state().numpy().tobytes()
                ).hexdigest(),
                protocol="Original safe_state followed by Python, NumPy and torch.manual_seed override",
            ),
            indent=2,
        )
    )
    logger = engine.get_logger(str(output))
    dataset = model.extract(args)
    engine.training(
        dataset,
        optimizer.extract(args),
        pipeline.extract(args),
        Path(args.source_path).name,
        [30000],
        sorted(set(args.save_iterations + [30000])),
        logger=logger,
    )
    visible = engine.render_sets(dataset, -1, pipeline.extract(args), logger=logger)
    engine.evaluate(str(output), visible_count=visible, logger=logger)


if __name__ == "__main__":
    main()
