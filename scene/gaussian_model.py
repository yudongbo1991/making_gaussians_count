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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation


class GaussianModel:

    def setup_functions(self):

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(
        self,
        feat_dim: int = 32,
        n_offsets: int = 5,
        voxel_size: float = 0.01,
        update_depth: int = 3,
        update_init_factor: int = 100,
        update_hierachy_factor: int = 4,
        use_feat_bank: bool = False,
        appearance_dim: int = 32,
        ratio: int = 1,
        add_opacity_dist: bool = False,
        add_cov_dist: bool = False,
        add_color_dist: bool = False,
    ):
        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.slot_keep = None
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank
        self.appearance_dim = appearance_dim
        self.embedding_appearance = None
        self.ratio = ratio
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist
        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        self.opacity_accum = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)
        self.anchor_demon = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim + 3 + self.opacity_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh(),
        ).cuda()
        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim + 3 + self.cov_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7 * self.n_offsets),
        ).cuda()
        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.mlp_color = nn.Sequential(
            nn.Linear(feat_dim + 3 + self.color_dist_dim + 0, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3 * self.n_offsets),
            nn.Sigmoid(),
        ).cuda()

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()

    @property
    def get_scaling(self):
        return 1.0 * self.scaling_activation(self._scaling)

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        return self._anchor

    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data / voxel_size), axis=0) * voxel_size
        return data

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        points = pcd.points[:: self.ratio]
        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0] * 0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()
        print(f"Initial voxel_size: {self.voxel_size}")
        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 1e-07)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        )
        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros(self.get_anchor.shape[0], device="cuda")

    def create_from_surface_prior(self, path, spatial_lr_scale, use_offsets):
        if use_offsets:
            raise ValueError("Final occupancy initialization uses zero offsets")
        prior = np.load(path)
        if not np.isclose(float(prior["voxel"]), self.voxel_size, rtol=1e-06):
            raise ValueError("Prior voxel and model voxel differ")
        anchors = prior["anchors"].copy()
        pcd = BasicPointCloud(
            points=anchors.copy(), colors=np.zeros_like(anchors), normals=np.zeros_like(anchors)
        )
        self.create_from_pcd(pcd, spatial_lr_scale)
        if not np.allclose(self._anchor.detach().cpu().numpy(), anchors, atol=1e-07):
            raise ValueError("Prior anchor ordering changed during voxelization")
        print(
            "Surface initialization:",
            len(anchors),
            "anchors; prior offsets:",
            use_offsets,
            flush=True,
        )

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.offset_gradient_accum = torch.zeros(
            (self.get_anchor.shape[0] * self.n_offsets, 1), device="cuda"
        )
        self.offset_gradient_squared_accum = torch.zeros_like(self.offset_gradient_accum)
        self.offset_denom = torch.zeros(
            (self.get_anchor.shape[0] * self.n_offsets, 1), device="cuda"
        )
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        l = [
            {
                "params": [self._anchor],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "anchor",
            },
            {
                "params": [self._offset],
                "lr": training_args.offset_lr_init * self.spatial_lr_scale,
                "name": "offset",
            },
            {"params": [self._anchor_feat], "lr": training_args.feature_lr, "name": "anchor_feat"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
            {
                "params": self.mlp_opacity.parameters(),
                "lr": training_args.mlp_opacity_lr_init,
                "name": "mlp_opacity",
            },
            {
                "params": self.mlp_cov.parameters(),
                "lr": training_args.mlp_cov_lr_init,
                "name": "mlp_cov",
            },
            {
                "params": self.mlp_color.parameters(),
                "lr": training_args.mlp_color_lr_init,
                "name": "mlp_color",
            },
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )
        self.offset_scheduler_args = get_expon_lr_func(
            lr_init=training_args.offset_lr_init * self.spatial_lr_scale,
            lr_final=training_args.offset_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.offset_lr_delay_mult,
            max_steps=training_args.offset_lr_max_steps,
        )
        self.mlp_opacity_scheduler_args = get_expon_lr_func(
            lr_init=training_args.mlp_opacity_lr_init,
            lr_final=training_args.mlp_opacity_lr_final,
            lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
            max_steps=training_args.mlp_opacity_lr_max_steps,
        )
        self.mlp_cov_scheduler_args = get_expon_lr_func(
            lr_init=training_args.mlp_cov_lr_init,
            lr_final=training_args.mlp_cov_lr_final,
            lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
            max_steps=training_args.mlp_cov_lr_max_steps,
        )
        self.mlp_color_scheduler_args = get_expon_lr_func(
            lr_init=training_args.mlp_color_lr_init,
            lr_final=training_args.mlp_color_lr_final,
            lr_delay_mult=training_args.mlp_color_lr_delay_mult,
            max_steps=training_args.mlp_color_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group["lr"] = lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(self._offset.shape[1] * self._offset.shape[2]):
            l.append("f_offset_{}".format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append("f_anchor_feat_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = (
            self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]
        packed_slots = None
        if self.slot_keep is not None:
            assert self.slot_keep.shape == (len(anchor), self.n_offsets)
            packed_slots = np.packbits(self.slot_keep.detach().cpu().numpy(), axis=1)
            dtype_full += [("slot_bits_" + str(i), "u1") for i in range(packed_slots.shape[1])]
        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1
        )
        if packed_slots is not None:
            attributes = np.concatenate((attributes, packed_slots), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)
        names = [p.name for p in plydata.elements[0].properties if p.name.startswith("slot_bits_")]
        self.slot_keep = None
        if names:
            expected = ["slot_bits_" + str(i) for i in range((self.n_offsets + 7) // 8)]
            if set(names) != set(expected):
                raise ValueError("Incomplete packed slot mask")
            packed = np.stack(
                [np.asarray(plydata.elements[0][name], dtype=np.uint8) for name in expected], axis=1
            )
            bits = np.unpackbits(packed, axis=1)
            if bits[:, self.n_offsets :].any():
                raise ValueError("Nonzero unused mask bits")
            self.slot_keep = torch.tensor(bits[:, : self.n_offsets].astype(bool), device="cuda")
        anchor = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        ).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)
        scale_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        anchor_feat_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")
        ]
        anchor_feat_names = sorted(anchor_feat_names, key=lambda x: int(x.split("_")[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offset_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")
        ]
        offset_names = sorted(offset_names, key=lambda x: int(x.split("_")[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        self._anchor_feat = nn.Parameter(
            torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._offset = nn.Parameter(
            torch.tensor(offsets, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if (
                "mlp" in group["name"]
                or "conv" in group["name"]
                or "feat_base" in group["name"]
                or ("embedding" in group["name"])
            ):
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def training_statis(
        self,
        viewspace_point_tensor,
        opacity,
        update_filter,
        offset_selection_mask,
        anchor_visible_mask,
    ):
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity < 0] = 0
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        self.anchor_demon[anchor_visible_mask] += 1
        anchor_visible_mask = (
            anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        )
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(
            dim=1
        )
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_gradient_squared_accum[combined_mask] += grad_norm.square()
        self.offset_denom[combined_mask] += 1

    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if (
                "mlp" in group["name"]
                or "conv" in group["name"]
                or "feat_base" in group["name"]
                or ("embedding" in group["name"])
            ):
                continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
                if group["name"] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:, 3:]
                    temp[temp > 0.05] = 0.05
                    group["params"][0][:, 3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group["name"] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:, 3:]
                    temp[temp > 0.05] = 0.05
                    group["params"][0][:, 3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_anchor(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)
        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    def anchor_growing(self, grads, threshold, offset_mask):
        from scene.multiscale_growth import grow

        return grow(self, grads, threshold, offset_mask)

    def adjust_anchor(
        self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005
    ):
        grads = self.offset_gradient_accum / self.offset_denom
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        grads_norm = (
            (self.offset_gradient_squared_accum / self.offset_denom.clamp_min(1))
            .clamp_min(0)
            .sqrt()
            .squeeze(1)
        )
        if not torch.isfinite(grads_norm).all():
            raise RuntimeError("Nonfinite cross-view RMS demand")
        offset_mask = (self.offset_denom > check_interval * success_threshold * 0.5).squeeze(dim=1)
        self.growth_demand_diagnostics = dict(
            squared_stat_rows=len(self.offset_gradient_squared_accum),
            eligible_offsets=int(offset_mask.sum()),
            mean_gradient_sum=float(grads.squeeze(1)[offset_mask].sum()),
            rms_gradient_sum=float(grads_norm[offset_mask].sum()),
        )
        self.anchor_growing(grads_norm, grad_threshold, offset_mask)
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros(
            [self.get_anchor.shape[0] * self.n_offsets - self.offset_denom.shape[0], 1],
            dtype=torch.int32,
            device=self.offset_denom.device,
        )
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)
        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros(
            [self.get_anchor.shape[0] * self.n_offsets - self.offset_gradient_accum.shape[0], 1],
            dtype=torch.int32,
            device=self.offset_gradient_accum.device,
        )
        self.offset_gradient_accum = torch.cat(
            [self.offset_gradient_accum, padding_offset_gradient_accum], dim=0
        )
        self.offset_gradient_squared_accum[offset_mask] = 0
        self.offset_gradient_squared_accum = torch.cat(
            [self.offset_gradient_squared_accum, torch.zeros_like(padding_offset_gradient_accum)],
            dim=0,
        )
        prune_mask = (self.opacity_accum < min_opacity * self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval * success_threshold).squeeze(dim=1)
        prune_mask = torch.logical_and(prune_mask, anchors_mask)
        self.offset_gradient_squared_accum = self.offset_gradient_squared_accum.view(
            -1, self.n_offsets
        )[~prune_mask].reshape(-1, 1)
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom
        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        if anchors_mask.sum() > 0:
            self.opacity_accum[anchors_mask] = torch.zeros(
                [anchors_mask.sum(), 1], device="cuda"
            ).float()
            self.anchor_demon[anchors_mask] = torch.zeros(
                [anchors_mask.sum(), 1], device="cuda"
            ).float()
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum
        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon
        if prune_mask.shape[0] > 0:
            self.prune_anchor(prune_mask)
        self.max_radii2D = torch.zeros(self.get_anchor.shape[0], device="cuda")

    def save_mlp_checkpoints(self, path, mode="split"):
        mkdir_p(os.path.dirname(path))
        self.mlp_opacity.eval()
        opacity_mlp = torch.jit.trace(
            self.mlp_opacity, torch.rand(1, self.feat_dim + 3 + self.opacity_dist_dim).cuda()
        )
        opacity_mlp.save(os.path.join(path, "opacity_mlp.pt"))
        self.mlp_opacity.train()
        self.mlp_cov.eval()
        cov_mlp = torch.jit.trace(
            self.mlp_cov, torch.rand(1, self.feat_dim + 3 + self.cov_dist_dim).cuda()
        )
        cov_mlp.save(os.path.join(path, "cov_mlp.pt"))
        self.mlp_cov.train()
        self.mlp_color.eval()
        color_mlp = torch.jit.trace(
            self.mlp_color, torch.rand(1, self.feat_dim + 3 + self.color_dist_dim + 0).cuda()
        )
        color_mlp.save(os.path.join(path, "color_mlp.pt"))
        self.mlp_color.train()

    def load_mlp_checkpoints(self, path, mode="split"):
        self.mlp_opacity = torch.jit.load(os.path.join(path, "opacity_mlp.pt")).cuda()
        self.mlp_cov = torch.jit.load(os.path.join(path, "cov_mlp.pt")).cuda()
        self.mlp_color = torch.jit.load(os.path.join(path, "color_mlp.pt")).cuda()
