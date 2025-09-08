from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import PlayerDetectionDataset, DataConfig, split_by_clip_groups
from .transforms import get_train_transforms, get_val_transforms


@dataclass
class AugmentationConfig:
    p_horizontal_flip: float = 0.5
    p_affine: float = 0.5
    scale_limit: float = 0.1
    shift_limit: float = 0.1
    p_blur: float = 0.2
    blur_limit: int = 5
    p_color_jitter: float = 0.5
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    saturation_limit: float = 0.2
    hue_limit: float = 0.1


@dataclass
class DataModuleConfig:
    images_root: str
    labeled_json: str
    val_ratio: float = 0.1
    split_seed: int = 42
    img_size: Tuple[int, int] = (640, 640)
    batch_size: int = 8
    num_workers: int = 4
    category_name: str = "player"
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


def _collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


class PlayerDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataModuleConfig):
        super().__init__()
        self.cfg = cfg
        self.train_dataset: Optional[PlayerDetectionDataset] = None
        self.val_dataset: Optional[PlayerDetectionDataset] = None

    def setup(self, stage: Optional[str] = None):
        height, width = self.cfg.img_size

        train_tf = get_train_transforms(height=height, width=width, **self.cfg.augmentation.__dict__)
        val_tf = get_val_transforms(height=height, width=width)

        base = {
            "images_root": self.cfg.images_root,
            "img_size": self.cfg.img_size,
            "category_name": self.cfg.category_name,
        }

        import json

        with open(self.cfg.labeled_json, "r") as f:
            data = json.load(f)

        train_ids, val_ids = split_by_clip_groups(data, self.cfg.val_ratio, self.cfg.split_seed)

        # Build datasets with the same underlying parsed JSON
        self.train_dataset = PlayerDetectionDataset(
            DataConfig(labeled_json=self.cfg.labeled_json, **base),
            transforms=train_tf,
            allowed_image_ids=train_ids,
            data_override=data,
        )
        self.val_dataset = PlayerDetectionDataset(
            DataConfig(labeled_json=self.cfg.labeled_json, **base),
            transforms=val_tf,
            allowed_image_ids=val_ids,
            data_override=data,
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
            collate_fn=_collate_fn,
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
            collate_fn=_collate_fn,
        )
