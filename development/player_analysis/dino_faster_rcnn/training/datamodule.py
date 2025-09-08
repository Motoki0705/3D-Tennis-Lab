from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import CocoDetDataset, detection_collate
from .transforms import get_train_transforms, get_val_transforms


class DetectionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_images: str,
        train_ann: str,
        val_images: Optional[str] = None,
        val_ann: Optional[str] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        image_size: int = 1024,
    ):
        super().__init__()
        self.train_images = train_images
        self.train_ann = train_ann
        self.val_images = val_images or train_images
        self.val_ann = val_ann or train_ann
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.image_size = image_size

    def setup(self, stage: Optional[str] = None):
        self.train_ds = CocoDetDataset(
            self.train_images, self.train_ann, transforms=get_train_transforms(self.image_size)
        )
        self.val_ds = CocoDetDataset(self.val_images, self.val_ann, transforms=get_val_transforms(self.image_size))

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=detection_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=detection_collate,
        )
