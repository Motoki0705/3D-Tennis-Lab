from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import random

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import BallVectorDataset, DataConfig


@dataclass
class DataModuleConfig:
    labeled_json: str
    sequence_length: int
    predict_offset: int
    val_ratio: float = 0.2
    split_seed: int = 42
    batch_size: int = 64
    num_workers: int = 4


class BallDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataModuleConfig):
        super().__init__()
        self.cfg = cfg
        self.train_dataset: Optional[BallVectorDataset] = None
        self.val_dataset: Optional[BallVectorDataset] = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            # Create a temporary full dataset to get clip info
            tmp_ds = BallVectorDataset(
                DataConfig(
                    labeled_json=self.cfg.labeled_json,
                    sequence_length=self.cfg.sequence_length,
                    predict_offset=self.cfg.predict_offset,
                )
            )
            num_clips = len(tmp_ds.clips)

            if num_clips < 2:
                train_idx, val_idx = list(range(num_clips)), []
            else:
                indices = list(range(num_clips))
                random.Random(self.cfg.split_seed).shuffle(indices)
                n_val = max(1, int(round(num_clips * self.cfg.val_ratio)))
                n_val = min(n_val, num_clips - 1)
                val_idx = sorted(indices[:n_val])
                train_idx = sorted(indices[n_val:])

            # Build actual datasets with the same underlying data to avoid re-parsing
            data_cfg = DataConfig(
                labeled_json=self.cfg.labeled_json,
                sequence_length=self.cfg.sequence_length,
                predict_offset=self.cfg.predict_offset,
            )
            self.train_dataset = BallVectorDataset(data_cfg, allowed_clip_indices=train_idx, data_override=tmp_ds.data)
            self.val_dataset = BallVectorDataset(data_cfg, allowed_clip_indices=val_idx, data_override=tmp_ds.data)

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not initialized. Call setup() first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized. Call setup() first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
        )
