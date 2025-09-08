from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple
import random

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import CocoDetectionDataset, DataConfig
from .transforms import get_train_transforms, get_val_transforms


@dataclass
class AugmentationConfig:
    p_horizontal_flip: float = 0.5
    p_affine: float = 0.4
    scale_limit: float = 0.1
    shift_limit: float = 0.1
    p_blur: float = 0.1
    blur_limit: int = 3
    p_color_jitter: float = 0.2
    brightness_limit: float = 0.1
    contrast_limit: float = 0.1
    saturation_limit: float = 0.05
    hue_limit: float = 0.03


@dataclass
class DataModuleConfig:
    images_root: str
    # Either single labeled_json (split by val_ratio) or explicit train/val jsons
    labeled_json: Optional[str] = None
    train_json: Optional[str] = None
    val_json: Optional[str] = None
    val_ratio: float = 0.1
    split_seed: int = 42
    img_size: Tuple[int, int] = (720, 1280)
    batch_size: int = 2
    num_workers: int = 4
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


class DetectionDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataModuleConfig):
        super().__init__()
        self.cfg = cfg
        self.train_dataset: Optional[CocoDetectionDataset] = None
        self.val_dataset: Optional[CocoDetectionDataset] = None

    def setup(self, stage: Optional[str] = None):
        height, width = self.cfg.img_size
        train_transforms = get_train_transforms(height=height, width=width, **self.cfg.augmentation.__dict__)
        val_transforms = get_val_transforms(height=height, width=width)

        if self.cfg.train_json and self.cfg.val_json:
            self.train_dataset = CocoDetectionDataset(
                DataConfig(images_root=self.cfg.images_root, ann_file=self.cfg.train_json, img_size=self.cfg.img_size),
                transforms=train_transforms,
            )
            self.val_dataset = CocoDetectionDataset(
                DataConfig(images_root=self.cfg.images_root, ann_file=self.cfg.val_json, img_size=self.cfg.img_size),
                transforms=val_transforms,
            )
            return

        if not self.cfg.labeled_json:
            raise RuntimeError("Either (train_json & val_json) or labeled_json must be provided")

        # Split a single json by images list
        tmp = CocoDetectionDataset(
            DataConfig(images_root=self.cfg.images_root, ann_file=self.cfg.labeled_json, img_size=self.cfg.img_size),
            transforms=None,
        )
        n = len(tmp.image_ids)
        indices = list(range(n))
        random.Random(int(self.cfg.split_seed)).shuffle(indices)
        n_val = max(1, int(round(n * float(self.cfg.val_ratio))))
        n_val = min(n_val, n - 1)
        val_idx = sorted(indices[:n_val])
        train_idx = sorted(indices[n_val:])

        self.train_dataset = CocoDetectionDataset(
            DataConfig(images_root=self.cfg.images_root, ann_file=self.cfg.labeled_json, img_size=self.cfg.img_size),
            transforms=train_transforms,
            keep_image_indices=train_idx,
            preloaded=tmp.preloaded,
        )
        self.val_dataset = CocoDetectionDataset(
            DataConfig(images_root=self.cfg.images_root, ann_file=self.cfg.labeled_json, img_size=self.cfg.img_size),
            transforms=val_transforms,
            keep_image_indices=val_idx,
            preloaded=tmp.preloaded,
        )

    @staticmethod
    def _collate(batch):
        images = [b[0] for b in batch]
        targets = [b[1] for b in batch]
        return images, targets

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
            collate_fn=self._collate,
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
            collate_fn=self._collate,
        )
