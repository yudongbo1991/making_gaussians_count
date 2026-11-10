"""Check analytic removal moments against pixelwise autograd on a tiny scene."""

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from method.analytic_risk import moments
from method.native import load_analyzers
from utils.graphics_utils import getProjectionMatrix


def main():
    xyz = torch.tensor([[0.0, 0.0, 1.0], [0.03, 0.0, 1.1], [-0.02, 0.01, 1.2]], device="cuda")
    opacity = torch.tensor([[0.25], [0.35], [0.5]], device="cuda", requires_grad=True)
    color = torch.tensor([[0.8, 0.1, 0.2], [0.1, 0.7, 0.3], [0.2, 0.3, 0.9]], device="cuda")
    scales = torch.full((3, 3), 0.15, device="cuda")
    rotation = torch.zeros(3, 4, device="cuda")
    rotation[:, 0] = 1
    settings = GaussianRasterizationSettings(
        image_height=9,
        image_width=11,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=torch.zeros(3, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, device="cuda"),
        projmatrix=getProjectionMatrix(0.01, 10.0, math.pi / 2, math.pi / 2).transpose(0, 1).cuda(),
        sh_degree=0,
        campos=torch.zeros(3, device="cuda"),
        prefiltered=False,
        debug=False,
    )
    rgb, radii = GaussianRasterizer(settings)(
        means3D=xyz,
        means2D=torch.zeros_like(xyz),
        shs=None,
        colors_precomp=color,
        opacities=opacity,
        scales=scales,
        rotations=rotation,
        cov3D_precomp=None,
    )
    ids = torch.tensor([0, 1, 10], device="cuda")
    slot, anchor, _ = moments(load_analyzers(), rgb, ids, 2, 10)
    expected_slot = torch.zeros(3, device="cuda", dtype=torch.float64)
    expected_anchor = torch.zeros(2, device="cuda", dtype=torch.float64)
    for pixel in range(rgb.numel()):
        derivative = torch.autograd.grad(
            rgb.clamp(0, 1).flatten()[pixel], opacity, retain_graph=True
        )[0]
        direction = (derivative * opacity.detach()).flatten().double()
        expected_slot += direction.square() / rgb.numel()
        grouped = torch.zeros(2, device="cuda", dtype=torch.float64).index_add_(
            0, ids // 10, direction
        )
        expected_anchor += grouped.square() / rgb.numel()
    torch.testing.assert_close(slot.double(), expected_slot, rtol=5e-4, atol=1e-9)
    torch.testing.assert_close(anchor.double(), expected_anchor, rtol=5e-4, atol=1e-9)
    assert (radii > 0).all()
    print("CUDA rendering and analytic slot/anchor moments: PASS")


if __name__ == "__main__":
    main()
