# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union
import torch
from nerfstudio.data.scene_box import OrientedBox
from copy import deepcopy
from nerfstudio.cameras.rays import RayBundle
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure,MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.cameras.cameras import Cameras
from gsplat._torch_impl import quat_to_rotmat
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.base_model import Model, ModelConfig
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
import viser.transforms as vtf
from nerfstudio.model_components.losses import scale_gauss_gradients_by_distance_squared
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig

from torchmetrics.image import StructuralSimilarityIndexMeasure
from gsplat.rasterize import RasterizeGaussians
from gsplat.project_gaussians import ProjectGaussians
from nerfstudio.model_components.losses import depth_ranking_loss
from gsplat.sh import SphericalHarmonics, num_sh_bases
from gsplat.nd_rasterize import NDRasterizeGaussians

from nerfstudio.fields.gaussian_lerf_field import GaussianLERFField
from nerfstudio.encoders.image_encoder import BaseImageEncoderConfig, BaseImageEncoder
from nerfstudio.field_components.gaussian_lerf_fieldheadnames import GaussianLERFFieldHeadNames
from nerfstudio.viewer_beta.viewer_elements import ViewerButton, ViewerSlider, ViewerControl, ViewerVec3
from nerfstudio.utils.colormaps import apply_colormap

def random_quat_tensor(N, **kwargs):
    u = torch.rand(N, **kwargs)
    v = torch.rand(N, **kwargs)
    w = torch.rand(N, **kwargs)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def projection_matrix(znear, zfar, fovx, fovy, device="cpu"):
    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )


@dataclass
class GaussianSplattingModelConfig(ModelConfig):
    """Gaussian Splatting Model Config"""

    _target: Type = field(default_factory=lambda: GaussianSplattingModel)
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 250
    """training starts at 1/d resolution, every n steps this is doubled"""
    num_downscales: int = 0
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians"""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling gaussians"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0002
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 1.0
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 1.0
    """if a gaussian is more than this percent of screen space, split it"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    extra_points: int = 0
    """number of extra points to add to the model in addition to sfm points, randomly distributed"""
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    sh_degree: int = 4
    """maximum degree of spherical harmonics to use"""
    clip_loss_weight: float = 0.1
    """weight of clip loss"""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig(mode="off")
    """camera optimizer config"""


class GaussianSplattingModel(Model):
    """Gaussian Splatting model

    Args:
        config: Gaussian Splatting configuration to instantiate model
    """

    config: GaussianSplattingModelConfig

    def __init__(self, *args, **kwargs):
        if "seed_points" in kwargs:
            self.seed_pts = kwargs["seed_points"]
        else:
            self.seed_pts = None
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        if self.seed_pts is not None and not self.config.random_init:
            extra_means = (torch.rand((self.config.extra_points, 3)) - 0.5) * 10
            self.means = torch.nn.Parameter(torch.cat([self.seed_pts[0],extra_means])) # (Location, Color)
        else:
            self.means = torch.nn.Parameter((torch.rand((1000000, 3)) - 0.5) * 10)
        self.xys_grad_norm = None
        self.max_2Dsize = None
        distances, _ = self.k_nearest_sklearn(self.means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        self.scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        self.quats = torch.nn.Parameter(random_quat_tensor(self.num_points))
        dim_sh = num_sh_bases(self.config.sh_degree)

        self.gaussian_lerf_field = GaussianLERFField()
        self.datamanager = self.kwargs["datamanager"]
        self.image_encoder: BaseImageEncoder = self.kwargs["image_encoder"]

        if self.seed_pts is not None and not self.config.random_init:
            fused_color = RGB2SH(self.seed_pts[1] / 255)
            shs = torch.zeros((fused_color.shape[0], 3, dim_sh)).float().cuda()
            shs[:, :3, 0] = fused_color
            shs[:, 3:, 1:] = 0.0
            #concat on extra_points amount of random ones at the end
            extra_shs = torch.rand((self.config.extra_points, 3, dim_sh)).float().cuda()
            extra_shs[:,:,1:] = 0.0#zero out the higher freq
            shs = torch.cat([shs, extra_shs])
            self.colors = torch.nn.Parameter(shs[:, :, 0:1])
            self.shs_rest = torch.nn.Parameter(shs[:, :, 1:])
        else:
            self.colors = torch.nn.Parameter(torch.rand(self.num_points, 3, 1))
            self.shs_rest = torch.nn.Parameter(torch.zeros((self.num_points, 3, dim_sh - 1)))

        self.opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(self.num_points, 1)))

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure()
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0
        
        self.crop_box: Optional[OrientedBox] = None
        self.back_color = torch.zeros(3)
        self.viewer_control = ViewerControl()
        self.viser_scale_ratio = 0.1

        self.crop_to_word = ViewerButton("Crop gaussians to word", cb_hook=self.crop_to_word_cb)
        self.reset_crop = ViewerButton("Reset crop", cb_hook=self.reset_crop_cb)
        self.crop_scale = ViewerSlider("Crop scale", 0.1, 0, 3.0, 0.01)
        self.relevancy_thresh = ViewerSlider("Relevancy Thresh", 0.5, 0, 1.0, 0.01)

        self._crop_center_init = None
        self.crop_ids = None#torch.ones_like(self.means[:,0],dtype=torch.bool)
        self.dropout_mask = None
        self.original_means = None

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

    @property
    def get_colors(self):
        color = self.colors
        shs_rest = self.shs_rest
        return torch.cat((color, shs_rest), dim=2)

    def load_state_dict(self, dict, **kwargs):
        # resize the parameters to match the new number of points
        self.step = 30000
        newp = dict["means"].shape[0]
        self.means = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.scales = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.quats = torch.nn.Parameter(torch.zeros(newp, 4, device=self.device))
        self.colors = torch.nn.Parameter(torch.zeros(self.num_points, 3, 1, device=self.device))
        self.shs_rest = torch.nn.Parameter(
            torch.zeros(self.num_points, 3, num_sh_bases(self.config.sh_degree) - 1, device=self.device)
        )
        self.opacities = torch.nn.Parameter(torch.zeros(newp, 1, device=self.device))
        super().load_state_dict(dict, **kwargs)

    def k_nearest_sklearn(self, x: torch.Tensor, k: int, include_self: bool = False):
        """
        Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        if include_self:
            return distances.astype(np.float32), indices
        else:
            return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
        param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2, reset_momentum=True):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))

        if reset_momentum:
            param_state["exp_avg"] = torch.cat(
                [param_state["exp_avg"], torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims)],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
        else:
            param_state['exp_avg'] = torch.cat([param_state['exp_avg'], param_state['exp_avg'][dup_mask.squeeze()].repeat(*repeat_dims)],dim=0)
            param_state['exp_avg_sq'] = torch.cat([param_state['exp_avg_sq'], param_state['exp_avg_sq'][dup_mask.squeeze()].repeat(*repeat_dims)],dim=0)

        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def after_train(self, step):
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            grads = self.xys.grad.detach().norm(dim=-1)  # TODO fill in
            opacity_grads = self.opacities.grad.detach().norm(dim=-1)
            #Save the grads to a text file for logging
            with open('grads.txt', 'a') as f:
                f.write(f"Step {self.step}, xys grad {opacity_grads.min().item()}, max {opacity_grads.max().item()}, mean {opacity_grads.mean().item()}\n")

            # print(f"Step {self.step}, xys grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = grads
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                self.vis_counts[visible_mask] = self.vis_counts[visible_mask] + 1
                self.xys_grad_norm[visible_mask] = grads[visible_mask] + self.xys_grad_norm[visible_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii,dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(self.max_2Dsize[visible_mask], newradii / float(max(self.last_size[0], self.last_size[1])))

    def set_crop(self, crop_box: OrientedBox):
        self.crop_box = crop_box

    def set_background(self, back_color: torch.Tensor):
        assert back_color.shape == (3,)
        self.back_color = back_color

    def refinement_after(self, optimizers: Optimizers, step):
        if self.step >= self.config.warmup_length:
            with torch.no_grad():
                # only split/cull if we've seen every image since opacity reset
                reset_interval = self.config.reset_alpha_every * self.config.refine_every
                if self.step < self.config.stop_split_at and self.step % reset_interval > self.num_train_data + self.config.refine_every:
                    # then we densify
                    avg_grad_norm = (
                        (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])
                    )
                    high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()
                    splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                    splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()
                    splits &= high_grads
                    nsamps = self.config.n_split_samples
                    (
                        split_means,
                        split_colors,
                        split_shs,
                        split_opacities,
                        split_scales,
                        split_quats,
                    ) = self.split_gaussians(splits, nsamps)

                    dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
                    dups &= high_grads
                    dup_means, dup_colors, dup_shs, dup_opacities, dup_scales, dup_quats = self.dup_gaussians(dups)
                    self.means = Parameter(torch.cat([self.means.detach(), split_means, dup_means], dim=0))
                    self.colors = Parameter(torch.cat([self.colors.detach(), split_colors, dup_colors], dim=0))
                    self.shs_rest = Parameter(torch.cat([self.shs_rest.detach(), split_shs, dup_shs], dim=0))
                    self.opacities = Parameter(
                        torch.cat([self.opacities.detach(), split_opacities, dup_opacities], dim=0)
                    )
                    self.scales = Parameter(torch.cat([self.scales.detach(), split_scales, dup_scales], dim=0))
                    self.quats = Parameter(torch.cat([self.quats.detach(), split_quats, dup_quats], dim=0))
                    # append zeros to the max_2Dsize tensor
                    self.max_2Dsize = torch.cat(
                        [self.max_2Dsize, torch.zeros_like(split_scales[:, 0]), torch.zeros_like(dup_scales[:, 0])],
                        dim=0,
                    )
                    split_idcs = torch.where(splits)[0]
                    param_groups = self.get_gaussian_param_groups()
                    for group,param in param_groups.items():
                        if group == 'lerf':
                            continue
                        self.dup_in_optim(optimizers.optimizers[group],split_idcs,param,n=nsamps, reset_momentum=group!="clip_hash")
                    dup_idcs = torch.where(dups)[0]
                    
                    param_groups = self.get_gaussian_param_groups()
                    for group,param in param_groups.items():
                        if group == 'lerf':
                            continue
                        self.dup_in_optim(optimizers.optimizers[group],dup_idcs,param,1)

                # Offset all the opacity reset logic by refine_every so that we don't
                # save checkpoints right when the opacity is reset (saves every 2k)
                if self.step % reset_interval > self.num_train_data + self.config.refine_every:
                    # then cull
                    deleted_mask = self.cull_gaussians()
                    param_groups = self.get_gaussian_param_groups()
                    for group, param in param_groups.items():
                        if group == 'lerf':
                            continue
                        self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)

                if self.step % reset_interval == self.config.refine_every:
                    reset_value = self.config.cull_alpha_thresh * 0.8
                    self.opacities.data = torch.full_like(
                        self.opacities.data, torch.logit(torch.tensor(reset_value)).item()
                    )
                    # reset the exp of optimizer
                    optim = optimizers.optimizers["opacity"]
                    param = optim.param_groups[0]["params"][0]
                    param_state = optim.state[param]
                    param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                    param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
                self.xys_grad_norm = None
                self.vis_counts = None
                self.max_2Dsize = None

    def cull_gaussians(self):
        """
        This function deletes gaussians with under a certain opacity threshold
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (torch.sigmoid(self.opacities) < self.config.cull_alpha_thresh).squeeze()
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            culls = culls | toobigs
            if self.step < self.config.stop_split_at:
                # cull big screen space
                culls = culls | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
        self.means = Parameter(self.means[~culls].detach())
        self.scales = Parameter(self.scales[~culls].detach())
        self.quats = Parameter(self.quats[~culls].detach())
        self.colors = Parameter(self.colors[~culls].detach())
        self.shs_rest = Parameter(self.shs_rest[~culls].detach())
        self.opacities = Parameter(self.opacities[~culls].detach())

        print(f"Culled {n_bef - self.num_points} gaussians")
        return culls

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """

        n_splits = split_mask.sum().item()
        print(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        new_colors = self.colors[split_mask].repeat(samps, 1, 1)
        # step 3, sample new shs
        new_shs = self.shs_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        return new_means, new_colors, new_shs, new_opacities, new_scales, new_quats

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        print(f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        dup_means = self.means[dup_mask]
        dup_colors = self.colors[dup_mask]
        dup_shs = self.shs_rest[dup_mask]
        dup_opacities = self.opacities[dup_mask]
        dup_scales = self.scales[dup_mask]
        dup_quats = self.quats[dup_mask]
        return dup_means, dup_colors, dup_shs, dup_opacities, dup_scales, dup_quats

    @property
    def num_points(self):
        return self.means.shape[0]

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers],
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            "xyz": [self.means],
            "color": [self.colors],
            "shs": [self.shs_rest],
            "opacity": [self.opacities],
            "scaling": [self.scales],
            "rotation": [self.quats],
            "lerf" : list(self.gaussian_lerf_field.parameters())
        }
    
    def crop_to_word_cb(self,element):
        with torch.no_grad():
            # clip_feats = self.gaussian_lerf_field.get_outputs_from_feature(self.clip_hash / self.clip_hash.norm(dim=-1,keepdim=True), self.crop_scale.value * torch.ones(self.num_points, 1, device=self.device))[GaussianLERFFieldHeadNames.CLIP].to(dtype=torch.float32)
            # clip_feats = self.gaussian_lerf_field.get_outputs(self.means, self.crop_scale.value * torch.ones(self.num_points, 1, device=self.device))[GaussianLERFFieldHeadNames.CLIP].to(dtype=torch.float32)
            # clip_feats = self.gaussian_lerf_field.get_outputs(self.means, self.best_scales[0].to(self.device) * torch.ones(self.num_points, 1, device=self.device))[GaussianLERFFieldHeadNames.CLIP].to(dtype=torch.float32)

            # Do K nearest neighbors for each point and then avg the clip hash for each point based on the KNN
            distances, indicies = self.k_nearest_sklearn(self.means.data, 3, True)
            distances = torch.from_numpy(distances).to(self.device)
            indicies = torch.from_numpy(indicies).to(self.device).view(-1)
            weights = torch.sigmoid(self.opacities[indicies].view(-1, 4))
            weights = torch.nn.Softmax(dim=-1)(weights)
            points = self.means[indicies]
            # clip_hash_encoding = self.gaussian_lerf_field.get_hash(self.means)
            clip_hash_encoding = self.gaussian_lerf_field.get_hash(points)
            clip_hash_encoding = clip_hash_encoding.view(-1, 4, clip_hash_encoding.shape[1])
            clip_hash_encoding = (clip_hash_encoding * weights.unsqueeze(-1))
            clip_hash_encoding = clip_hash_encoding.sum(dim=1)
            clip_feats = self.gaussian_lerf_field.get_outputs_from_feature(clip_hash_encoding, self.best_scales[0].to(self.device) * torch.ones(self.num_points, 1, device=self.device))[GaussianLERFFieldHeadNames.CLIP].to(dtype=torch.float32)
            relevancy = self.image_encoder.get_relevancy(clip_feats / (clip_feats.norm(dim=-1, keepdim=True)+1e-6), 0).view(self.num_points, -1)
            color = apply_colormap(relevancy[..., 0:1])
            self.viewer_control.viser_server.add_point_cloud("relevancy", self.means.numpy(force=True) * 10, color.numpy(force=True), 0.01)

            # Add a slider to debug the relevancy values
            
            # self.crop_ids = (relevancy[..., 0] > self.relevancy_thresh.value)
            
            #Define all crop viewer elements
            self.crop_points = relevancy[..., 0] > self.relevancy_thresh.value
            self._crop_center_init = self.means[self.crop_points].mean(dim=0).cpu().numpy()
            self.original_means = self.means.data.clone()

            self._crop_handle = self.viewer_control.viser_server.add_transform_controls("Crop Points", depth_test=False, line_width=4.0)
            world_center = tuple(p / self.viser_scale_ratio for p in self._crop_center_init)
            self._crop_handle.position = world_center

            @self._crop_handle.on_update
            def _update_crop_handle(han):
                # import pdb; pdb.set_trace()
                if self._crop_center_init is None:
                    return
                # import pdb; pdb.set_trace()
                new_center = np.array(self._crop_handle.position) * self.viser_scale_ratio
                delta = new_center - self._crop_center_init
                displacement = torch.zeros_like(self.means)
                displacement[self.crop_points] = torch.from_numpy(delta).to(self.device).to(self.means.dtype)
                
                curr_to_world = torch.from_numpy(vtf.SE3(np.concatenate((self._crop_handle.wxyz, self._crop_handle.position * self.viser_scale_ratio))).as_matrix()).to(self.device).to(self.means.dtype)
                transform = torch.from_numpy(vtf.SE3(np.concatenate((self._crop_handle.wxyz, (self._crop_handle.position * self.viser_scale_ratio) - self._crop_center_init))).as_matrix()).to(self.device).to(self.means.dtype)

                print(f"transform {transform}")
                transformed_points = self.original_means.clone()
                homogeneous_points = torch.cat((transformed_points[self.crop_points], torch.ones(transformed_points[self.crop_points].shape[0], 1, device=self.device, dtype=self.means.dtype)), dim=1)
                transformed_homogeneous = curr_to_world @ transform @ torch.inverse(curr_to_world) @ homogeneous_points.transpose(0,1)
                transformed_homogeneous = transformed_homogeneous.transpose(0,1)
                transformed_points[self.crop_points] = transformed_homogeneous[:, :3] / transformed_homogeneous[:, 3:4]
                self.means.data = transformed_points

            # self._crop_center.value = tuple(p / self.viser_scale_ratio for p in self._crop_center_init)

            self.viewer_control.viser_server.add_point_cloud("Centroid", self._crop_center_init / self.viser_scale_ratio, np.array([0,0,0]), 0.1)

    def reset_crop_cb(self,element):
        self.crop_ids = None#torch.ones_like(self.means[:,0],dtype=torch.bool)
        self.means.data = self.original_means
        self._crop_center_init = None
        self._crop_handle.visible = False
        
    def get_max_across(self, xys, depths, radii, conics, num_tiles_hit, opacities, h, w, preset_scales=None):
        # probably not a good idea bc it's prob going to be a lot of memory
        n_phrases = len(self.image_encoder.positives)
        n_phrases_maxs = [None for _ in range(n_phrases)]
        n_phrases_sims = [None for _ in range(n_phrases)]
        scales_list = torch.linspace(0.0, 1.5, 30).to(self.device)
        # scales_list = [0.1]
        all_probs = []

        with torch.no_grad():
            clip_hash_encoding = self.gaussian_lerf_field.get_hash(self.means)
            clip_output = NDRasterizeGaussians.apply(
                            xys,
                            depths,
                            radii,
                            conics,
                            num_tiles_hit,
                            # self.gaussian_lerf_field.get_outputs_from_feature(self.clip_hash / self.clip_hash.norm(dim=-1, keepdim=True), scale * torch.ones(h*w, 1, device=self.device))[GaussianLERFFieldHeadNames.CLIP].view(self.num_points, -1).to(dtype=torch.float32),
                            # self.clip_hash / self.clip_hash.norm(dim=-1, keepdim=True),
                            # clip_hash_encoding / clip_hash_encoding.norm(dim=-1, keepdim=True),
                            clip_hash_encoding,
                            opacities,
                            h,
                            w,
                            # torch.zeros(self.image_encoder.embedding_dim, device=self.device),
                            torch.zeros(clip_hash_encoding.shape[1], device=self.device),
                        )
            # Normalize the clip output
            # clip_output = clip_output / (clip_output.norm(dim=-1, keepdim=True) + 1e-6)
        for i, scale in enumerate(scales_list):
            with torch.no_grad():
                clip_output_im = self.gaussian_lerf_field.get_outputs_from_feature(clip_output.view(h*w, -1), scale * torch.ones(h*w, 1, device=self.device))[GaussianLERFFieldHeadNames.CLIP].to(dtype=torch.float32).view(h, w, -1)

            for j in range(n_phrases):
                if preset_scales is None or j == i:
                    # relevancy_rasterized = NDRasterizeGaussians.apply(
                    #         xys,
                    #         depths,
                    #         radii,
                    #         conics,
                    #         num_tiles_hit,
                    #         self.image_encoder.get_relevancy(self.gaussian_lerf_field.get_outputs_from_feature(self.clip_hash / self.clip_hash.norm(dim=-1, keepdim=True), scale * torch.ones(self.num_points, 1, device=self.device))[GaussianLERFFieldHeadNames.CLIP].view(self.num_points, -1).to(dtype=torch.float32), j)[..., 0:1],
                    #         opacities,
                    #         h,
                    #         w,
                    #         torch.zeros(1, device=self.device),
                    #     )
                    
                    probs = self.image_encoder.get_relevancy(clip_output_im.view(-1, self.image_encoder.embedding_dim), j)
                    pos_prob = probs[..., 0:1]
                    all_probs.append((pos_prob.max(), scale))
                    if n_phrases_maxs[j] is None or pos_prob.max() > n_phrases_sims[j].max():
                        n_phrases_maxs[j] = scale
                        n_phrases_sims[j] = pos_prob
        # print(f"Best scales: {n_phrases_maxs}")#, Words: {self.image_encoder.positives}, Scale List: {scales_list}, All probs: {all_probs}")
        # import pdb; pdb.set_trace()
        return torch.stack(n_phrases_sims), torch.Tensor(n_phrases_maxs)#, relevancy_rasterized


    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        # add camera optimizer param groups
        self.camera_optimizer.get_param_groups(gps)
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max((self.config.num_downscales - self.step // self.config.resolution_schedule), 0)
        else:
            return 1

    def project_gaussians(self, camera, downscale_factor=1):
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"
        if self.training:
            #currently relies on the branch vickie/camera-grads
            self.camera_optimizer.apply_to_camera(camera)
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return {"rgb": torch.full((camera.height.item(), camera.width.item(), 3), 0.5, device=self.device)}
        else:
            crop_ids = None
        camera_downscale = downscale_factor
        camera.rescale_output_resolution(1 / camera_downscale)
        # shift the camera to center of scene looking at center
        R = camera.camera_to_worlds[0, :3, :3]  # 3 x 3
        T = camera.camera_to_worlds[0, :3, 3:4]  # 3 x 1
        # flip the z axis to align with gsplat conventions
        R_edit = torch.tensor(vtf.SO3.from_x_radians(np.pi).as_matrix(), device=R.device, dtype=R.dtype)
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        fovx = 2 * math.atan(camera.width / (2 * camera.fx))
        fovy = 2 * math.atan(camera.height / (2 * camera.fy))
        W, H = camera.width.item(), camera.height.item()
        self.last_size = (H, W)
        projmat = projection_matrix(0.001, 1000, fovx, fovy, device=self.device)
        BLOCK_X, BLOCK_Y = 16, 16
        tile_bounds = (
            (W + BLOCK_X - 1) // BLOCK_X,
            (H + BLOCK_Y - 1) // BLOCK_Y,
            1,
        )

        if crop_ids is not None:
            means_crop = self.means[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            means_crop = self.means
            scales_crop = self.scales
            quats_crop = self.quats
        xys, depths, radii, conics, num_tiles_hit, cov3d = ProjectGaussians.apply(
            means_crop,
            torch.exp(scales_crop),
            1,
            quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            projmat.squeeze() @ viewmat.squeeze(),
            camera.fx.item(),
            camera.fy.item(),
            cx,
            cy,
            H,
            W,
            tile_bounds,
        )

        # rescale the camera back to original dimensions
        camera.rescale_output_resolution(camera_downscale)
        return xys, depths, radii, conics, num_tiles_hit, cov3d, W, H

    # @profile
    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"
        outputs = {}
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return {"rgb": torch.full((camera.height.item(), camera.width.item(), 3), 0.5, device=self.device)}
        else:
            crop_ids = None
        camera_downscale = self._get_downscale_factor()

        self.xys, depths, self.radii, conics, num_tiles_hit, cov3d, W, H = self.project_gaussians(camera, downscale_factor=camera_downscale)
        # if self.training:
        #     # experimental gradient scaling for floater removal
        #     depths_de = depths.detach()
        #     opacities_crop = scale_gauss_gradients_by_distance_squared(opacities_crop,depths_de,far_dist=1,padding_eps=.3)
        # Important to allow xys grads to populate properly
        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            colors_crop = self.get_colors[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            colors_crop = self.get_colors
        if self.training:
            self.xys.retain_grad()
            background = torch.rand(3, device=self.device)
        else:
            background = self.back_color
        if self.config.sh_degree > 0:
            viewdirs = means_crop.detach() - camera.camera_to_worlds.detach()[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            n_coeffs = num_sh_bases(n)
            coeffs = colors_crop.permute(0, 2, 1)[:, :n_coeffs, :]
            rgbs = SphericalHarmonics.apply(n, viewdirs, coeffs)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = self.get_colors.squeeze()  # (N, 3)
            rgbs = torch.sigmoid(rgbs)
        rgb = RasterizeGaussians.apply(
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,
            rgbs,
            torch.sigmoid(opacities_crop),
            H,
            W,
            background,
        )
        outputs["rgb"] = rgb
        depth_im = None
        depth_im = RasterizeGaussians.apply(
            self.xys.detach(),
            depths,
            self.radii,
            conics.detach(),
            num_tiles_hit,
            depths[:, None].repeat(1, 3),
            torch.sigmoid(opacities_crop.detach()),
            H,
            W,
            torch.ones(3, device=self.device) * 10,
        )[..., 0:1]
        outputs["depth"] = depth_im

        reset_interval = self.config.reset_alpha_every * self.config.refine_every
        if self.training and self.step>self.config.warmup_length and (self.step % reset_interval > self.num_train_data + self.config.refine_every  or self.step < (self.config.reset_alpha_every * self.config.refine_every)):
            outputs["clip"] = None
            return outputs
            with torch.no_grad():
                clip_xys, clip_depths, clip_radii, clip_conics, clip_num_tiles_hit, clip_cov3d, clip_W, clip_H = self.project_gaussians(camera, downscale_factor=camera.metadata["clip_downscale_factor"])
            # clip_H = H//camera.metadata["clip_downscale_factor"]
            # clip_W = W//camera.metadata["clip_downscale_factor"]
            #Very messy will fix to get it from camera metadata
            self.random_pixels = self.datamanager.random_pixels.to(self.device)
            clip_scale = self.datamanager.curr_scale * torch.ones((self.random_pixels.shape[0],1),device=self.device)
            clip_scale = clip_scale * clip_H * (depth_im.view(-1, 1)[self.random_pixels] / camera.fy.item())
            # print("Current scale: ", self.datamanager.curr_scale, "Clip scale mean: ", clip_scale.mean(), "Clip scale max: ", clip_scale.max(), "Clip scale min: ", clip_scale.min())
            # import pdb; pdb.set_trace()
            clip_hash_encoding = self.gaussian_lerf_field.get_hash(self.means)

            field_output = NDRasterizeGaussians.apply(
                clip_xys.detach(),
                clip_depths.detach(),
                clip_radii.detach(),
                clip_conics.detach(),
                clip_num_tiles_hit,
                # clip_hash_encoding[self.dropout_mask] / clip_hash_encoding[self.dropout_mask].norm(dim=-1, keepdim=True),
                # clip_hash_encoding[self.dropout_mask],
                clip_hash_encoding,
                torch.sigmoid(opacities_crop.detach().clone()),
                clip_H,
                clip_W,
                torch.zeros(clip_hash_encoding.shape[1], device=self.device),
            )
            # field_output = NDRasterizeGaussians.apply(
            #     self.xys.detach(),
            #     depths.detach(),
            #     self.radii.detach(),
            #     conics.detach(),
            #     num_tiles_hit,
            #     # clip_hash_encoding[self.dropout_mask] / clip_hash_encoding[self.dropout_mask].norm(dim=-1, keepdim=True),
            #     # clip_hash_encoding[self.dropout_mask],
            #     clip_hash_encoding,
            #     torch.sigmoid(opacities_crop.detach()),
            #     clip_H,
            #     clip_W,
            #     torch.zeros(clip_hash_encoding.shape[1], device=self.device),
            # )
            # Normalize the clip output
            # clip_output = clip_output / (clip_output.norm(dim=-1, keepdim=True) + 1e-6)
            field_output = self.gaussian_lerf_field.get_outputs_from_feature(field_output.view(clip_H*clip_W, -1)[self.random_pixels], clip_scale)
            clip_output = field_output[GaussianLERFFieldHeadNames.CLIP].to(dtype=torch.float32)
            # dino_output = field_output[GaussianLERFFieldHeadNames.DINO].to(dtype=torch.float32)

            # import pdb; pdb.set_trace()
            outputs["clip"] = clip_output
            outputs["clip_scale"] = clip_scale
            # outputs["dino"] = dino_output
        if not self.training:
                    # N x B x 1; N
                    max_across, self.best_scales = self.get_max_across(
                        self.xys,
                        depths,
                        self.radii,
                        conics,
                        num_tiles_hit,
                        torch.sigmoid(self.opacities[crop_ids]),
                        H,
                        W,
                    )
                    # import pdb; pdb.set_trace()
                    for i in range(len(self.image_encoder.positives)):
                        max_across[i][max_across[i] < self.relevancy_thresh.value] = 0
                        # relevancy_rasterized[relevancy_rasterized < 0.5] = 0
                        outputs[f"relevancy_{i}"] = max_across[i].view(H, W, -1)
                        # outputs[f"relevancy_rasterized_{i}"] = relevancy_rasterized.view(H, W, -1)
                        # outputs[f"best_scales_{i}"] = best_scales[i]
                
        return outputs

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        d = self._get_downscale_factor()
        if d > 1:
            # use torchvision to resize
            import torchvision.transforms.functional as TF

            newsize = (batch["image"].shape[0] // d, batch["image"].shape[1] // d)
            gt_img = TF.resize(batch["image"].permute(2, 0, 1), newsize).permute(1, 2, 0)
        else:
            gt_img = batch["image"]
        metrics_dict = {}
        gt_rgb = gt_img.to(self.device)  # RGB or RGBA image
        # gt_rgb = self.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        metrics_dict['gaussian_count'] = self.num_points
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        loss_dict = {}
        d = self._get_downscale_factor()
        if d > 1:
            # use torchvision to resize
            import torchvision.transforms.functional as TF

            newsize = (batch["image"].shape[0] // d, batch["image"].shape[1] // d)
            gt_img = TF.resize(batch["image"].permute(2, 0, 1), newsize).permute(1, 2, 0)
        else:
            gt_img = batch['image']
        Ll1 = torch.abs(gt_img- outputs['rgb']).mean()
        simloss = (1-self.ssim(gt_img.permute(2,0,1)[None,...], outputs['rgb'].permute(2,0,1)[None,...]))
        loss_dict["main_loss"] = (1-self.config.ssim_lambda)*Ll1 + self.config.ssim_lambda*simloss
        
        if self.training and 'clip' in outputs: 
            # import pdb; pdb.set_trace()
            unreduced_clip = self.config.clip_loss_weight * torch.nn.functional.huber_loss(
                outputs["clip"], batch["clip"].to(torch.float32), delta=1.25, reduction="none"
            )
            loss_dict["clip_loss"] = unreduced_clip.sum(dim=-1).nanmean()
            
        if self.training and 'dino' in outputs:
            unreduced_dino = torch.nn.functional.mse_loss(outputs["dino"], batch["dino"], reduction="none")
            loss_dict["dino_loss"] = unreduced_dino.sum(dim=-1).nanmean()

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(
        self, camera_ray_bundle: RayBundle, camera: Optional[Cameras] = None
    ) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        assert camera is not None, "must provide camera to gaussian model"
        outs = self.get_outputs(camera.to(self.device))
        return outs

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict
