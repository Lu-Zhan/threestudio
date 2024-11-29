from dataclasses import dataclass, field
from functools import partial

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
# from threestudio.models.estimators import ImportanceEstimator
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.networks import create_network_with_input_encoding
from threestudio.models.renderers.base import VolumeRenderer
from threestudio.systems.utils import parse_optimizer, parse_scheduler_to_instance
from threestudio.utils.ops import chunk_batch, get_activation, validate_empty_rays
from threestudio.utils.typing import *


@threestudio.register("voxel-renderer")
class VoxelRenderer(VolumeRenderer):
    @dataclass
    class Config(VolumeRenderer.Config):
        num_samples_per_ray: int = 512
        eval_chunk_size: int = 160000
        randomized: bool = True

        near_plane: float = 0.0
        far_plane: float = 1e10

        return_comp_normal: bool = False
        return_normal_perturb: bool = False

        # in ["occgrid", "proposal", "importance"]
        estimator: str = "occgrid"

        # for occgrid
        grid_prune: bool = True
        prune_alpha_threshold: bool = True

        # for proposal
        proposal_network_config: Optional[dict] = None
        prop_optimizer_config: Optional[dict] = None
        prop_scheduler_config: Optional[dict] = None
        num_samples_per_ray_proposal: int = 64

        # for importance
        num_samples_per_ray_importance: int = 64

    cfg: Config

    def configure(
        self,
        video_shape: Tuple[int, int, int, int],
    ) -> None:
        # super().configure(geometry, material, background)
        # if self.cfg.estimator == "occgrid":
        #     self.estimator = nerfacc.OccGridEstimator(
        #         roi_aabb=self.bbox.view(-1), resolution=32, levels=1
        #     )
        #     if not self.cfg.grid_prune:
        #         self.estimator.occs.fill_(True)
        #         self.estimator.binaries.fill_(True)
        #     self.render_step_size = (
        #         1.732 * 2 * self.cfg.radius / self.cfg.num_samples_per_ray
        #     )
        #     self.randomized = self.cfg.randomized
        # elif self.cfg.estimator == "importance":
        #     self.estimator = ImportanceEstimator()
        # elif self.cfg.estimator == "proposal":
        #     self.prop_net = create_network_with_input_encoding(
        #         **self.cfg.proposal_network_config
        #     )
        #     self.prop_optim = parse_optimizer(
        #         self.cfg.prop_optimizer_config, self.prop_net
        #     )
        #     self.prop_scheduler = (
        #         parse_scheduler_to_instance(
        #             self.cfg.prop_scheduler_config, self.prop_optim
        #         )
        #         if self.cfg.prop_scheduler_config is not None
        #         else None
        #     )
        #     self.estimator = nerfacc.PropNetEstimator(
        #         self.prop_optim, self.prop_scheduler
        #     )

        #     def get_proposal_requires_grad_fn(
        #         target: float = 5.0, num_steps: int = 1000
        #     ):
        #         schedule = lambda s: min(s / num_steps, 1.0) * target

        #         steps_since_last_grad = 0

        #         def proposal_requires_grad_fn(step: int) -> bool:
        #             nonlocal steps_since_last_grad
        #             target_steps_since_last_grad = schedule(step)
        #             requires_grad = steps_since_last_grad > target_steps_since_last_grad
        #             if requires_grad:
        #                 steps_since_last_grad = 0
        #             steps_since_last_grad += 1
        #             return requires_grad

        #         return proposal_requires_grad_fn

        #     self.proposal_requires_grad_fn = get_proposal_requires_grad_fn()
        #     self.randomized = self.cfg.randomized
        # else:
        #     raise NotImplementedError(
        #         "Unknown estimator, should be one of ['occgrid', 'proposal', 'importance']."
        #     )

        # for proposal
        # self.vars_in_forward = {}

        self.voxels = torch.nn.Parameter(
            torch.randn(video_shape) * 0.5 + 0.5, requires_grad=True
        )

    def forward(
        self,
        t_index: torch.Long,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:
        
        frame = self.voxels[t_index] # (c, h, w)

        out = {
            "comp_rgb": frame,
        )

        return out

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        if self.cfg.estimator == "occgrid":
            if self.cfg.grid_prune:

                def occ_eval_fn(x):
                    density = self.geometry.forward_density(x)
                    # approximate for 1 - torch.exp(-density * self.render_step_size) based on taylor series
                    return density * self.render_step_size

                if self.training and not on_load_weights:
                    self.estimator.update_every_n_steps(
                        step=global_step, occ_eval_fn=occ_eval_fn
                    )
        elif self.cfg.estimator == "proposal":
            if self.training:
                requires_grad = self.proposal_requires_grad_fn(global_step)
                self.vars_in_forward["requires_grad"] = requires_grad
            else:
                self.vars_in_forward["requires_grad"] = False

    def update_step_end(self, epoch: int, global_step: int) -> None:
        if self.cfg.estimator == "proposal" and self.training:
            self.estimator.update_every_n_steps(
                self.vars_in_forward["trans"],
                self.vars_in_forward["requires_grad"],
                loss_scaler=1.0,
            )

    def train(self, mode=True):
        self.randomized = mode and self.cfg.randomized
        if self.cfg.estimator == "proposal":
            self.prop_net.train()
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        if self.cfg.estimator == "proposal":
            self.prop_net.eval()
        return super().eval()
