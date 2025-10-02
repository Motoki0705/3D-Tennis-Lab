from __future__ import annotations

from typing import Any, Mapping, Sequence
import copy

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

try:  # Optional dependency when running under Hydra/OmegaConf.
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover - OmegaConf not installed
    DictConfig = ()  # type: ignore
    OmegaConf = None  # type: ignore


_SENTINEL = object()


class BaseDataModule(pl.LightningDataModule):
    """Generic DataModule that owns one canonical dataset and splits it.

    - Accepts optional per-split transforms and applies them by cloning the
      base dataset per split to avoid transform collisions across Subsets.
    - Reads dataloader parameters from either `dataloader` or `dataset` section
      in the provided config-like object.
    - Reads split ratios and optional seed from `splits` (fallback to
      `dataset.train_ratio`, etc.).
    """

    def __init__(
        self,
        config,
        dataset,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        collate_fn=None,
    ) -> None:
        super().__init__()
        try:
            self.save_hyperparameters(config)
        except Exception:
            pass
        self.config = config
        self._dataset_cfg = self._get_section(config, "dataset")
        self._dataloader_cfg = self._get_section(config, "dataloader")
        self._splits_cfg = self._get_section(config, "splits")

        self.full_dataset = dataset
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.collate_fn = collate_fn

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Any = None) -> None:
        total = len(self.full_dataset)
        n_train, n_val, n_test = self._resolve_split_lengths(total)

        # Deterministic split indices (supports optional seed in splits config)
        seed = self._get_value(self._splits_cfg, "seed", 42)
        try:
            seed = int(seed)
        except Exception:
            seed = 42
        indices = list(range(total))
        import random as _random

        _random.Random(seed).shuffle(indices)
        i_train = indices[:n_train]
        i_val = indices[n_train : n_train + n_val]
        i_test = indices[n_train + n_val : n_train + n_val + n_test]

        # Clone base dataset per split to allow distinct transforms
        base = self.full_dataset
        train_base = copy.deepcopy(base)
        val_base = copy.deepcopy(base)
        test_base = copy.deepcopy(base)

        if self.train_transforms:
            self._assign_transform(train_base, self.train_transforms)
        if self.val_transforms:
            self._assign_transform(val_base, self.val_transforms)
        if self.test_transforms:
            self._assign_transform(test_base, self.test_transforms)

        self.train_dataset = Subset(train_base, i_train)
        self.val_dataset = Subset(val_base, i_val)
        self.test_dataset = Subset(test_base, i_test)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self._get_loader_value("batch_size", default=1),
            shuffle=True,
            num_workers=self._get_loader_value("num_workers", default=0),
            pin_memory=self._get_loader_value("pin_memory", default=False),
            persistent_workers=self._get_loader_value("persistent_workers", default=False),
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self._get_loader_value("batch_size", default=1),
            num_workers=self._get_loader_value("num_workers", default=0),
            pin_memory=self._get_loader_value("pin_memory", default=False),
            persistent_workers=self._get_loader_value("persistent_workers", default=False),
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self._get_loader_value("batch_size", default=1),
            num_workers=self._get_loader_value("num_workers", default=0),
            pin_memory=self._get_loader_value("pin_memory", default=False),
            persistent_workers=self._get_loader_value("persistent_workers", default=False),
            collate_fn=self.collate_fn,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assign_transform(self, dataset, transform) -> None:
        if hasattr(dataset, "transform"):
            dataset.transform = transform

    def _get_loader_value(self, key: str, *, default: Any) -> Any:
        value = self._get_value(self._dataloader_cfg, key, default=_SENTINEL)
        if value is not _SENTINEL:
            return value
        return self._get_value(self._dataset_cfg, key, default)

    def _resolve_split_lengths(self, total: int) -> Sequence[int]:
        splits = self._splits_cfg
        train_ratio = self._get_value(splits, "train_ratio", None) or self._get_value(splits, "train", None)
        val_ratio = self._get_value(splits, "val_ratio", None) or self._get_value(splits, "val", None)
        test_ratio = self._get_value(splits, "test_ratio", None) or self._get_value(splits, "test", None)

        # Backwards compatibility with older configs storing ratios under dataset.*
        if train_ratio is None and val_ratio is None and test_ratio is None:
            train_ratio = self._get_value(self._dataset_cfg, "train_ratio", None)
            val_ratio = self._get_value(self._dataset_cfg, "val_ratio", None)
            test_ratio = self._get_value(self._dataset_cfg, "test_ratio", None)

        # Default ratios when nothing supplied.
        if train_ratio is None and val_ratio is None and test_ratio is None:
            train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1

        train_ratio = float(train_ratio if train_ratio is not None else 0.0)
        val_ratio = float(val_ratio if val_ratio is not None else 0.0)
        if test_ratio is None:
            test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
        test_ratio = float(test_ratio)

        return self._ratios_to_lengths(total, [train_ratio, val_ratio, test_ratio])

    def _ratios_to_lengths(self, total: int, ratios: Sequence[float]) -> Sequence[int]:
        if total <= 0:
            return [0 for _ in ratios]
        positive = [max(0.0, float(r)) for r in ratios]
        denom = sum(positive)
        if denom <= 0.0:
            lengths = [0 for _ in ratios]
            lengths[0] = total
            return lengths
        normalized = [r / denom for r in positive]
        lengths = [int(round(total * r)) for r in normalized]
        diff = total - sum(lengths)
        while diff != 0:
            if diff > 0:
                idx = max(range(len(lengths)), key=lambda i: normalized[i])
                lengths[idx] += 1
                diff -= 1
            else:
                idx = max(range(len(lengths)), key=lambda i: lengths[i])
                if lengths[idx] == 0:
                    break
                lengths[idx] -= 1
                diff += 1
        return lengths

    def _get_section(self, cfg: Any, key: str):
        container = self._to_container(cfg)
        if isinstance(container, Mapping):
            return container.get(key)
        return getattr(cfg, key, None)

    def _get_value(self, section: Any, key: str, default: Any):
        if section is None:
            return default
        if isinstance(section, Mapping):
            return section.get(key, default)
        return getattr(section, key, default)

    def _to_container(self, cfg: Any):
        if OmegaConf is not None and isinstance(cfg, DictConfig):
            return OmegaConf.to_container(cfg, resolve=True)
        return cfg
