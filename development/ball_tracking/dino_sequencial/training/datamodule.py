from __future__ import annotations

import multiprocessing as mp
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from .dataset import BallSequenceDataset, DatasetConfig
from .resolution import ProgressiveResolutionConfig, ProgressiveResolutionController
from .sampler import ProgressiveResolutionSampler


@dataclass
class DataModuleConfig:
    images_root: str
    labeled_json: str
    img_size: Tuple[int, int] = (640, 640)
    output_stride: int = 4
    sigma_px: float = 2.0
    sequence_length: int = 8
    batch_size: int = 4  # Smaller batch size for sequences
    num_workers: int = 4
    val_ratio: float = 0.1
    seed: int = 42
    progressive_resolution: Optional[ProgressiveResolutionConfig] = None
    prefetch_factor: int = 1

    @classmethod
    def from_config(cls, cfg_like: Any) -> "DataModuleConfig":
        cfg_dict = _to_dict(cfg_like)
        if "cfg" in cfg_dict and isinstance(cfg_dict["cfg"], (Mapping, DictConfig)):
            cfg_dict = _to_dict(cfg_dict["cfg"])
        progressive_cfg: Optional[ProgressiveResolutionConfig] = None
        if "progressive_resolution" in cfg_dict:
            progressive_cfg_dict = _to_dict(cfg_dict.get("progressive_resolution", {}))
            if progressive_cfg_dict:
                progressive_cfg = ProgressiveResolutionConfig.from_mapping(progressive_cfg_dict)
        return cls(
            images_root=abspath(str(cfg_dict.get("images_root", "data/images"))),
            labeled_json=abspath(str(cfg_dict.get("labeled_json", "data/annotations.json"))),
            img_size=tuple(cfg_dict.get("img_size", (640, 640))),
            output_stride=int(cfg_dict.get("output_stride", 4)),
            sigma_px=float(cfg_dict.get("sigma_px", 2.0)),
            sequence_length=int(cfg_dict.get("sequence_length", 8)),
            batch_size=int(cfg_dict.get("batch_size", 4)),
            num_workers=int(cfg_dict.get("num_workers", 4)),
            val_ratio=float(cfg_dict.get("val_ratio", 0.1)),
            seed=int(cfg_dict.get("seed", cfg_dict.get("split_seed", 42))),
            progressive_resolution=progressive_cfg,
            prefetch_factor=int(cfg_dict.get("prefetch_factor", 1)),
        )


class BallSequenceDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataModuleConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_set = None
        self.val_set = None
        self.train_dataset: Optional[BallSequenceDataset] = None
        self.val_dataset: Optional[BallSequenceDataset] = None
        self._train_resolution: Optional[ProgressiveResolutionController] = None
        self._train_sampler: Optional[ProgressiveResolutionSampler] = None
        self._mp_manager: Optional[mp.Manager] = None
        self._shared_long_side_map = None
        self._prefetch_factor_value: Optional[int] = None

    def setup(self, stage: str | None = None) -> None:
        ds_cfg = DatasetConfig(
            images_root=self.cfg.images_root,
            labeled_json=self.cfg.labeled_json,
            img_size=self.cfg.img_size,
            output_stride=self.cfg.output_stride,
            sigma_px=self.cfg.sigma_px,
            sequence_length=self.cfg.sequence_length,
        )
        progressive_cfg = self.cfg.progressive_resolution
        self._train_resolution = None
        if progressive_cfg and progressive_cfg.enabled:
            self._train_resolution = ProgressiveResolutionController(progressive_cfg)
        train_dataset = BallSequenceDataset(ds_cfg)
        val_dataset = BallSequenceDataset(ds_cfg)

        self._prefetch_factor_value = self._compute_prefetch_factor()

        n = len(train_dataset)
        n_val = int(round(n * self.cfg.val_ratio))
        rng = np.random.default_rng(self.cfg.seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_set = Subset(train_dataset, train_idx.tolist())
        self.val_set = Subset(val_dataset, val_idx.tolist())
        if self._train_resolution is not None:
            if self._mp_manager is None:
                self._mp_manager = mp.Manager()
            self._shared_long_side_map = self._mp_manager.dict()  # type: ignore[assignment]
            train_dataset.set_shared_long_side_map(self._shared_long_side_map)
            self._train_sampler = ProgressiveResolutionSampler(
                subset_size=len(train_idx),
                subset_indices=train_idx.tolist(),
                batch_size=self.cfg.batch_size,
                controller=self._train_resolution,
                drop_last=False,
                seed=self.cfg.seed,
            )
            # initialise epoch 0 schedule
            self._train_resolution.on_epoch_start(0)
            self._train_sampler.set_epoch(0)
            self.train_dataset.set_long_side_map(self._train_sampler.index_long_side)
        else:
            self._train_sampler = None

    def train_dataloader(self) -> DataLoader:
        if self.train_set is None:
            raise RuntimeError("DataModule.setup must be called before requesting train_dataloader().")
        loader_kwargs = {
            "batch_size": self.cfg.batch_size,
            "shuffle": False if self._train_sampler is not None else True,
            "num_workers": self.cfg.num_workers,
            "pin_memory": True,
        }
        if self._train_sampler is not None:
            loader_kwargs["sampler"] = self._train_sampler
        if self._prefetch_factor_value is not None:
            loader_kwargs["prefetch_factor"] = self._prefetch_factor_value
        return DataLoader(self.train_set, **loader_kwargs)

    @property
    def uses_progressive_resolution(self) -> bool:
        return self._train_resolution is not None and self._train_resolution.enabled

    def set_progressive_max_epochs(self, max_epochs: int) -> None:
        if self._train_resolution is not None:
            self._train_resolution.set_max_epochs(max_epochs)

    def on_progressive_epoch_start(self, epoch: int) -> None:
        if self._train_resolution is None or self._train_sampler is None or self.train_dataset is None:
            return
        self._train_resolution.on_epoch_start(epoch)
        self._train_sampler.set_epoch(epoch)
        self.train_dataset.set_long_side_map(self._train_sampler.index_long_side)

    def val_dataloader(self) -> DataLoader:
        if self.val_set is None:
            raise RuntimeError("DataModule.setup must be called before requesting val_dataloader().")
        loader_kwargs = {
            "batch_size": self.cfg.batch_size,
            "shuffle": False,
            "num_workers": self.cfg.num_workers,
            "pin_memory": True,
        }
        if self._prefetch_factor_value is not None:
            loader_kwargs["prefetch_factor"] = self._prefetch_factor_value
        return DataLoader(self.val_set, **loader_kwargs)

    def _compute_prefetch_factor(self) -> Optional[int]:
        if self.cfg.num_workers <= 0:
            return None
        value = max(1, int(self.cfg.prefetch_factor))
        return value


def _to_dict(cfg_like: Any) -> Mapping[str, Any]:
    if isinstance(cfg_like, DictConfig):
        return OmegaConf.to_container(cfg_like, resolve=True)  # type: ignore[return-value]
    if isinstance(cfg_like, Mapping):
        return cfg_like
    return {}


def build_datamodule(cfg_like: Any) -> BallSequenceDataModule:
    dm_cfg = DataModuleConfig.from_config(cfg_like)
    return BallSequenceDataModule(dm_cfg)


__all__ = ["DataModuleConfig", "BallSequenceDataModule", "build_datamodule"]
