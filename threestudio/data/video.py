import bisect
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *


@dataclass
class VideoEventDataModuleConfig:
    # height and width should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 128
    width: Any = 128
    image_path: str = ""


class VideoEventDataset(Dataset):
    def __init__(self, voxels, mode) -> None:
        super().__init__()

        self.voxels = voxels
        self.num_frame = voxels.shape[0]

        self.mode = mode

    def __len__(self):
        return self.num_frame - 1

    def __getitem__(self, index):
        index_prev = index
        index_curr = index + 1

        image_prev = self.voxels[index_prev]
        image_curr = self.voxels[index_curr]

        return {
            "image_prev": image_prev,
            "image_curr": image_curr,
            "index_prev": index_prev,
            "index_curr": index_curr,
        }


@register("video-event-datamodule")
class VideoEventDataModule(pl.LightningDataModule):
    cfg: VideoEventDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(VideoEventDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = VideoEventDataset(voxels=self.voxels, mode="train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = VideoEventDataset(voxels=self.voxels, mode="val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = VideoEventDataset(voxels=self.voxels, mode="test")

    def prepare_data(self):
        data_dir = self.cfg.image_path

        # read frames
        self.voxels = None

    def general_loader(self, dataset, batch_size, shuffle=False) -> DataLoader:
        return DataLoader(
            dataset, num_workers=0, batch_size=batch_size, shuffle=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)
