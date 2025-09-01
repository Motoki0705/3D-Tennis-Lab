from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

from ...utils.lightning.base_datamodule import BaseDataModule
from ...utils.transformers.keypoint_transformer import prepare_transforms
from .dataset import BallDataset, HeatmapSpec, build_heatmap_specs


@dataclass
class VMixScheduleItem:
    epoch_le: int
    weights: Dict[str, float]


class BallDataModule(BaseDataModule):
    def __init__(self, config):
        # Build transforms
        train_tf, val_tf = prepare_transforms(img_size=config.dataset.img_size)

        # Build dataset
        specs: List[HeatmapSpec] = build_heatmap_specs(
            strides=list(config.dataset.heatmap.strides),
            sigmas=list(config.dataset.heatmap.sigmas),
        )

        # Unified dataset (sequence-aware). When length==1, acts like single-frame dataset.
        seq_cfg_obj = getattr(config.dataset, "sequence", None)

        def _seq_get(name, default=None):
            if seq_cfg_obj is None:
                return default
            if isinstance(seq_cfg_obj, dict):
                return seq_cfg_obj.get(name, default)
            return getattr(seq_cfg_obj, name, default)

        dataset = BallDataset(
            img_dir=config.dataset.img_dir,
            annotation_file=config.dataset.annotation_file,
            img_size=tuple(config.dataset.img_size),
            heatmap_specs=specs,
            sequence_length=int(_seq_get("length", 1)),
            frame_stride=int(_seq_get("frame_stride", 1)),
            center_version=str(_seq_get("center_version", None)) if _seq_get("center_version", None) else None,
            center_span=int(_seq_get("center_span", 1)),
            negatives=str(config.dataset.negatives),
            transform=None,  # set after split
        )

        super().__init__(
            config=config,
            dataset=dataset,
            train_transforms=train_tf,
            val_transforms=val_tf,
            test_transforms=val_tf,
        )

        # State for sampling
        self._current_version_weights = dict(config.dataset.version_weights)
        self._v_mix_schedule: List[VMixScheduleItem] = []
        for item in list(getattr(config.dataset, "v_mix_schedule", [])):
            weights = {k: float(v) for k, v in item.items() if k != "epoch_le"}
            self._v_mix_schedule.append(VMixScheduleItem(epoch_le=int(item["epoch_le"]), weights=weights))

    # Override to use split.* ratios from config
    def setup(self, stage=None):
        from torch.utils.data import random_split

        n_data = len(self.full_dataset)
        train_ratio = float(self.config.dataset.split.train_ratio)
        val_ratio = float(self.config.dataset.split.val_ratio)
        n_train = int(n_data * train_ratio)
        n_val = int(n_data * val_ratio)
        n_test = n_data - n_train - n_val

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.full_dataset, [n_train, n_val, n_test]
        )

        # Apply transforms
        if self.train_transforms:
            self.train_dataset.dataset.transform = self.train_transforms
        if self.val_transforms:
            self.val_dataset.dataset.transform = self.val_transforms
        if self.test_transforms:
            self.test_dataset.dataset.transform = self.test_transforms

    # Override to inject custom train sampler
    def train_dataloader(self):
        sampler_name: str = str(self.config.dataset.loader.sampler)
        batch_size = int(self.config.dataset.loader.batch_size)
        num_workers = int(self.config.dataset.loader.num_workers)
        pin_memory = bool(self.config.dataset.loader.pin_memory)
        persistent_workers = bool(self.config.dataset.loader.persistent_workers)

        if sampler_name == "uniform":
            sampler = RandomSampler(self.train_dataset)
        else:
            weights = self._build_sample_weights(self.train_dataset.indices)
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    def val_dataloader(self):
        batch_size = int(self.config.dataset.loader.batch_size)
        num_workers = int(self.config.dataset.loader.num_workers)
        pin_memory = bool(self.config.dataset.loader.pin_memory)
        persistent_workers = bool(self.config.dataset.loader.persistent_workers)
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    def test_dataloader(self):
        batch_size = int(self.config.dataset.loader.batch_size)
        num_workers = int(self.config.dataset.loader.num_workers)
        pin_memory = bool(self.config.dataset.loader.pin_memory)
        persistent_workers = bool(self.config.dataset.loader.persistent_workers)
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    def _build_sample_weights(self, subset_indices: List[int]) -> np.ndarray:
        policy: str = str(self.config.dataset.loader.sampler)

        # Obtain version labels from the underlying dataset
        ds: BallDataset = self.train_dataset.dataset  # type: ignore
        versions_all = ds.versions

        if policy == "balanced":
            # Equalize v1/v2 contributions
            counts: Dict[str, int] = {}
            for i in subset_indices:
                v = versions_all[i]
                counts[v] = counts.get(v, 0) + 1
            weights_map: Dict[str, float] = {}
            for v, c in counts.items():
                if c > 0:
                    weights_map[v] = 1.0 / c
            weights = np.array([weights_map.get(versions_all[i], 0.0) for i in subset_indices], dtype=np.float32)
            return weights

        # weighted policy with fallback for unknown versions
        vweights = self._current_version_weights
        known_vals = [float(v) for v in vweights.values()] or [1.0]
        default_eps = max(min(known_vals) * 0.1, 1e-3)
        weights = np.array(
            [float(vweights.get(versions_all[i], default_eps)) for i in subset_indices], dtype=np.float32
        )
        return weights

    # Call from LightningModule.on_train_epoch_start; requires Trainer(reload_dataloaders_every_n_epochs=1)
    def update_sampling_for_epoch(self, current_epoch: int):
        # Apply schedule if configured
        for item in self._v_mix_schedule:
            if current_epoch <= item.epoch_le:
                self._current_version_weights = dict(item.weights)
                return
        # Fall back to static weights
        self._current_version_weights = dict(self.config.dataset.version_weights)
