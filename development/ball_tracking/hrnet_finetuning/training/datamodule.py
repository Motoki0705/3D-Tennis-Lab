from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import random

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
    # Single annotation file; DataModule will split into train/val by clip
    labeled_json: str
    # Train/val split params when using a single file
    val_ratio: float = 0.1
    split_seed: int = 42
    img_size: Tuple[int, int] = (288, 512)
    T: int = 3
    frame_stride: int = 1
    output_stride: int = 1
    sigma_px: float = 3.0
    batch_size: int = 8
    num_workers: int = 4
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

        data_base = {
            "images_root": self.cfg.images_root,
            "img_size": self.cfg.img_size,
            "T": self.cfg.T,
            "frame_stride": self.cfg.frame_stride,
            "output_stride": self.cfg.output_stride,
            "sigma_px": self.cfg.sigma_px,
        }

        if stage == "fit" or stage is None:
            ann = self.cfg.labeled_json

            # Build a temporary full dataset to obtain clip grouping
            tmp_ds = BallDataset(DataConfig(labeled_json=ann, **data_base), transforms=None)
            num_clips = len(tmp_ds.clips)
            if num_clips < 2:
                # Fallback: use all clips for train and no val
                train_idx = list(range(num_clips))
                val_idx: List[int] = []
            else:
                indices = list(range(num_clips))
                random.Random(int(self.cfg.split_seed)).shuffle(indices)
                n_val = max(1, int(round(num_clips * float(self.cfg.val_ratio))))
                n_val = min(n_val, num_clips - 1)  # ensure at least one train clip
                val_idx = sorted(indices[:n_val])
                train_idx = sorted(indices[n_val:])

            # Build actual datasets with the same underlying data to avoid re-parsing
            self.train_dataset = BallDataset(
                DataConfig(labeled_json=ann, **data_base),
                transforms=train_transforms,
                allowed_clip_indices=train_idx,
                data_override=tmp_ds.data,
            )
            self.val_dataset = BallDataset(
                DataConfig(labeled_json=ann, **data_base),
                transforms=val_transforms,
                allowed_clip_indices=val_idx,
                data_override=tmp_ds.data,
            )

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
