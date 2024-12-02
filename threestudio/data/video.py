# import bisect
# import math
import h5py
import os
from dataclasses import dataclass, field

# import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

# import threestudio
from threestudio import register
# from threestudio.data.uncond import (
#     RandomCameraDataModuleConfig,
#     RandomCameraDataset,
#     RandomCameraIterableDataset,
# )
# from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
# from threestudio.utils.misc import get_rank
# from threestudio.utils.ops import (
#     get_mvp_matrix,
#     get_projection_matrix,
#     get_ray_directions,
#     get_rays,
# )
from threestudio.utils.typing import *


@dataclass
class VideoEventDataModuleConfig:
    # height and width should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 64
    width: Any = 64
    image_path: str = ""
    batch_size: int = 1


class VideoEventDataset(Dataset):
    def __init__(self, voxels_path, mode, num_frame=None) -> None:
        super().__init__()

        self.voxels = h5py.File(voxels_path, 'r')
        self.num_frame = self.voxels['images'].shape[0] if num_frame is None else num_frame

        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return 9999
        else:
            return self.num_frame - 1

    def __getitem__(self, index):
        index = index % (self.num_frame - 1)

        index_prev = index
        index_curr = index + 1

        image_prev = self.voxels['images'][index_prev]
        image_curr = self.voxels['images'][index_curr]

        image_prev = torch.from_numpy(np.array(image_prev).astype(np.float32))
        image_curr = torch.from_numpy(np.array(image_curr).astype(np.float32))

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
            self.train_dataset = VideoEventDataset(voxels_path=self.voxels_path, mode="train", num_frame=100)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = VideoEventDataset(voxels_path=self.voxels_path, mode="val", num_frame=100)
        if stage in [None, "test", "predict"]:
            self.test_dataset = VideoEventDataset(voxels_path=self.voxels_path, mode="test", num_frame=100)

    def prepare_data(self):
        data_dir = self.cfg.image_path

        # read h5 frames
        self.voxels_path = os.path.join(data_dir, "images.h5")

    def general_loader(self, dataset, batch_size, shuffle=False) -> DataLoader:
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)
