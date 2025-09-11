from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import CocoDetDataset, detection_collate
from .transforms import get_train_transforms, get_val_transforms, ResolutionMixTransform


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
        image_size_low: Optional[int] = None,
        image_size_high: Optional[int] = None,
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
        self.image_size_low = image_size_low
        self.image_size_high = image_size_high

        # Will be assigned in setup
        self._train_transform = None

    def setup(self, stage: Optional[str] = None):
        # Build training transforms: either mixed-resolution or single-size
        if self.image_size_low is not None and self.image_size_high is not None:
            self._train_transform = ResolutionMixTransform(
                low_size=self.image_size_low, high_size=self.image_size_high, p_high=0.0
            )
        else:
            self._train_transform = get_train_transforms(self.image_size)

        self.train_ds = CocoDetDataset(
            self.train_images,
            self.train_ann,
            transforms=self._train_transform,
            target_category="player",
            required_instances_per_image=2,
        )
        self.val_ds = CocoDetDataset(
            self.val_images,
            self.val_ann,
            transforms=get_val_transforms(self.image_size),
            target_category="player",
            required_instances_per_image=2,
        )

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

    # --- Utilities for callbacks ---
    def set_resolution_mix_ratio(self, p_high: float) -> None:
        """If mixed-resolution is enabled, update the mix ratio.

        Safe to call when single-size transforms are used (no-op).
        """
        if isinstance(self._train_transform, ResolutionMixTransform):
            self._train_transform.set_p_high(p_high)
