from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
import pytorch_lightning as pl
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

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

    @classmethod
    def from_config(cls, cfg_like: Any) -> "DataModuleConfig":
        cfg_dict = _to_dict(cfg_like)
        return cls(
            images_root=abspath(str(cfg_dict.get("images_root", "data/images"))),
            labeled_json=abspath(str(cfg_dict.get("labeled_json", "data/annotations.json"))),
            img_size=tuple(cfg_dict.get("img_size", (640, 640))),
            output_stride=int(cfg_dict.get("output_stride", 4)),
            sigma_px=float(cfg_dict.get("sigma_px", 2.0)),
            batch_size=int(cfg_dict.get("batch_size", 8)),
            num_workers=int(cfg_dict.get("num_workers", 4)),
            val_ratio=float(cfg_dict.get("val_ratio", 0.1)),
            seed=int(cfg_dict.get("split_seed", 42)),
        )


class BallHeatmapDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataModuleConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_set = None
        self.val_set = None

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
        if self.train_set is None:
            raise RuntimeError("DataModule.setup must be called before requesting train_dataloader().")
        return DataLoader(
            self.train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_set is None:
            raise RuntimeError("DataModule.setup must be called before requesting val_dataloader().")
        return DataLoader(
            self.val_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )


def _to_dict(cfg_like: Any) -> Mapping[str, Any]:
    if isinstance(cfg_like, DictConfig):
        return OmegaConf.to_container(cfg_like, resolve=True)  # type: ignore[return-value]
    if isinstance(cfg_like, Mapping):
        return cfg_like
    return {}


def build_datamodule(cfg_like: Any) -> BallHeatmapDataModule:
    dm_cfg = DataModuleConfig.from_config(cfg_like)
    return BallHeatmapDataModule(dm_cfg)


__all__ = ["DataModuleConfig", "BallHeatmapDataModule", "build_datamodule"]
