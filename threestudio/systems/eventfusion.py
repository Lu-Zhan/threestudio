from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("eventfusion-system")
class EventFusion(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        pass

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        # super().configure()
        video_shape = [100, 3, 512, 512]
        voxels = torch.sigmoid(torch.randn(video_shape))

        self.voxels = torch.nn.Parameter(
            voxels, requires_grad=True
        )
    
    def forward(self, t_index):
        return self.voxels[t_index] # (n, c, h, w)

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
    
    def _separate_batch(self, batch):
        pass
    
    def _calculate_event_loss(self, image_prev, image_curr, gt_diff):
        pred_diff = image_curr - image_prev
        return torch.nn.functional.mse_loss(pred_diff, gt_diff)

    def training_step(self, batch, batch_idx):
        gt_image_prev = batch['image_prev']
        gt_image_curr = batch['image_curr']
        index_prev = batch['index_prev']
        index_curr = batch['index_curr']

        noisy_image_prev = self(index_prev)
        noisy_image_curr = self(index_curr)

        prompt_utils = self.prompt_processor()

        guidance_out_prev = self.guidance(
            rgb=noisy_image_prev.permute(0, 2, 3, 1), 
            prompt_utils=prompt_utils,
            rgb_as_latents=False,
        )

        guidance_out_curr = self.guidance(
            rgb=noisy_image_curr.permute(0, 2, 3, 1), 
            prompt_utils=prompt_utils,
            rgb_as_latents=False,
        )

        loss_sds = (guidance_out_prev["loss_sds"] + guidance_out_curr["loss_sds"]) / 2
        self.log("train/loss_sds", loss_sds)

        denoised_prev_image = guidance_out_prev["denoised_image"]
        denoised_curr_image = guidance_out_curr["denoised_image"]

        loss_event = self._calculate_event_loss(
            image_prev=denoised_prev_image, 
            image_curr=denoised_curr_image, 
            gt_diff=(gt_image_curr - gt_image_prev),
        )
        self.log("train/loss_sds", loss_sds)

        loss = loss_sds + loss_event

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        batch_prev, _ = self._separate_batch(batch)
        image_prev = self(batch_prev['t_index'])

        self.save_image_grid(
            f"it{self.true_global_step}-{batch['t_index']}.png",
            [   
                {
                    "type": "rgb",
                    "img": batch_prev['gt_frame'][0],
                    "kwargs": {"data_format": "CHW"},
                },
                {
                    "type": "rgb",
                    "img": image_prev,
                    "kwargs": {"data_format": "CHW"},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        batch_prev, _ = self._separate_batch(batch)
        image_prev = self(batch_prev['t_index'])

        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['t_index']}.png",
            [   
                {
                    "type": "rgb",
                    "img": batch_prev['gt_frame'][0],
                    "kwargs": {"data_format": "CHW"},
                },
                {
                    "type": "rgb",
                    "img": image_prev,
                    "kwargs": {"data_format": "CHW"},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=25,
            name="test",
            step=self.true_global_step,
        )
