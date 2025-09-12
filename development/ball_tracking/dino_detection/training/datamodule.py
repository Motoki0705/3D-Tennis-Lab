from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

from .dataset import BallHeatmapDataset, DatasetConfig


@dataclass
class DataModuleConfig:
    images_root: str
    labeled_json: str
    img_size: Tuple[int, int] = (640, 640)
    output_stride: int = 4
    sigma_px: float = 2.0
    batch_size: int = 8
    num_workers: int = 4
    val_ratio: float = 0.1
    seed: int = 42


class BallHeatmapDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataModuleConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str | None = None) -> None:
        ds_cfg = DatasetConfig(
            images_root=self.cfg.images_root,
            labeled_json=self.cfg.labeled_json,
            img_size=self.cfg.img_size,
            output_stride=self.cfg.output_stride,
            sigma_px=self.cfg.sigma_px,
        )
        full = BallHeatmapDataset(ds_cfg)

        n = len(full)
        n_val = int(round(n * self.cfg.val_ratio))
        rng = np.random.default_rng(self.cfg.seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        self.train_set = Subset(full, train_idx.tolist())
        self.val_set = Subset(full, val_idx.tolist())

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )


__all__ = ["DataModuleConfig", "BallHeatmapDataModule"]
