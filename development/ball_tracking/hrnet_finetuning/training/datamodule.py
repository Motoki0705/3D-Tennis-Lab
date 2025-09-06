from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import BallDataset, DataConfig
from .transforms import get_train_transforms, get_val_transforms


@dataclass
class AugmentationConfig:
    p_horizontal_flip: float = 0.5
    p_affine: float = 0.5
    scale_limit: float = 0.1
    shift_limit: float = 0.1
    p_blur: float = 0.3
    blur_limit: int = 5
    p_color_jitter: float = 0.5
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    saturation_limit: float = 0.2
    hue_limit: float = 0.1


@dataclass
class DataModuleConfig:
    images_root: str
    labeled_json_train: str
    labeled_json_val: str
    img_size: Tuple[int, int]
    T: int
    frame_stride: int
    output_stride: int
    sigma_px: float
    batch_size: int
    num_workers: int
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


class BallDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataModuleConfig):
        super().__init__()
        self.cfg = cfg
        self.train_dataset: Optional[BallDataset] = None
        self.val_dataset: Optional[BallDataset] = None

    def setup(self, stage: Optional[str] = None):
        height, width = self.cfg.img_size

        train_transforms = get_train_transforms(height=height, width=width, **self.cfg.augmentation.__dict__)
        val_transforms = get_val_transforms(height=height, width=width)

        data_cfg = {
            "images_root": self.cfg.images_root,
            "img_size": self.cfg.img_size,
            "T": self.cfg.T,
            "frame_stride": self.cfg.frame_stride,
            "output_stride": self.cfg.output_stride,
            "sigma_px": self.cfg.sigma_px,
        }
        data_cfg_train = DataConfig(labeled_json=self.cfg.labeled_json_train, **data_cfg)
        data_cfg_val = DataConfig(labeled_json=self.cfg.labeled_json_val, **data_cfg)

        if stage == "fit" or stage is None:
            self.train_dataset = BallDataset(data_cfg_train, transforms=train_transforms)
            self.val_dataset = BallDataset(data_cfg_val, transforms=val_transforms)

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("Train dataset is not initialized. Call setup() first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset is not initialized. Call setup() first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
