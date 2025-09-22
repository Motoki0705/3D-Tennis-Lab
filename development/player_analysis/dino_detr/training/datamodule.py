from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader

try:
    from hydra.utils import to_absolute_path as hydra_to_absolute_path
except Exception:  # pragma: no cover - hydra not installed
    hydra_to_absolute_path = None

try:
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover - OmegaConf not installed
    DictConfig = ()  # type: ignore
    OmegaConf = None  # type: ignore

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


# ---------------------------------------------------------------------------
# Hydra-style factory helpers
# ---------------------------------------------------------------------------


def _to_dict(cfg_like: Any) -> Mapping[str, Any]:
    if cfg_like is None:
        return {}
    if OmegaConf is not None and isinstance(cfg_like, DictConfig):  # type: ignore[arg-type]
        return OmegaConf.to_container(cfg_like, resolve=True)  # type: ignore[return-value]
    if isinstance(cfg_like, Mapping):
        return cfg_like
    if hasattr(cfg_like, "__dict__"):
        return dict(vars(cfg_like))
    raise TypeError(f"Unsupported config type for DetectionDataModule: {type(cfg_like)!r}")


def _resolve_path(path: Optional[str]) -> Optional[str]:
    if path in (None, "", "null"):
        return None
    if hydra_to_absolute_path is not None:
        return hydra_to_absolute_path(str(path))
    return str(Path(str(path)).expanduser().resolve())


def _as_tuple(value, length: int = 2) -> Tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= length:
        return tuple(int(v) for v in value[:length])  # type: ignore[return-value]
    if isinstance(value, int):
        return (int(value), int(value))
    raise ValueError("img_size must be an int or sequence of two ints")


def build_datamodule(cfg_like: Any) -> DetectionDataModule:
    data_cfg = _to_dict(cfg_like)
    raw_images_root = data_cfg.get("images_root")
    images_root = _resolve_path(
        raw_images_root
        if isinstance(raw_images_root, str)
        else None
        if raw_images_root is None
        else str(raw_images_root)
    )
    if images_root is None:
        raise ValueError("'images_root' must be provided in the data config")

    def _optional_path(key: str) -> Optional[str]:
        value = data_cfg.get(key)
        if isinstance(value, Mapping):
            raise TypeError(f"{key} must be a path string, got mapping")
        if value is None:
            return None
        return _resolve_path(str(value))

    aug_cfg = data_cfg.get("augmentation", {})
    if not isinstance(aug_cfg, Mapping):
        raise TypeError("augmentation config must be a mapping")

    dm_cfg = DataModuleConfig(
        images_root=images_root,
        labeled_json=_optional_path("labeled_json"),
        train_json=_optional_path("train_json"),
        val_json=_optional_path("val_json"),
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        split_seed=int(data_cfg.get("split_seed", 42)),
        img_size=_as_tuple(data_cfg.get("img_size", (720, 1280))),
        batch_size=int(data_cfg.get("batch_size", 2)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        augmentation=AugmentationConfig(**{k: v for k, v in aug_cfg.items()}),
    )

    return DetectionDataModule(dm_cfg)


__all__ = ["DetectionDataModule", "DataModuleConfig", "build_datamodule", "AugmentationConfig"]
