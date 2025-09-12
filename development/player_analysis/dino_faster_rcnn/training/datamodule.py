from __future__ import annotations

from typing import Optional, Dict, Any, Iterable

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
        aspect_ratio: Optional[float | str] = None,
        target_category: str = "player",
        required_instances_per_image: int = 2,
        splits: Optional[Dict[str, Any]] = None,
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
        self._aspect_ratio = self._parse_aspect_ratio(aspect_ratio)
        self._target_category = target_category
        self._required_instances = int(required_instances_per_image)
        self._splits_cfg = splits

    @staticmethod
    def _parse_aspect_ratio(value: Optional[float | str]) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value) if value > 0 else None
        if isinstance(value, str):
            s = value.strip()
            if ":" in s:
                try:
                    w, h = s.split(":", 1)
                    w = float(w)
                    h = float(h)
                    if w > 0 and h > 0:
                        return w / h
                except Exception:
                    return None
            else:
                try:
                    f = float(s)
                    return f if f > 0 else None
                except Exception:
                    return None
        return None

    def setup(self, stage: Optional[str] = None):
        # Build training transforms: either mixed-resolution or single-size
        if self.image_size_low is not None and self.image_size_high is not None:
            self._train_transform = ResolutionMixTransform(
                low_size=self.image_size_low,
                high_size=self.image_size_high,
                p_high=0.0,
                aspect_ratio=self._aspect_ratio,
            )
        else:
            self._train_transform = get_train_transforms(self.image_size, aspect_ratio=self._aspect_ratio)

        if self._splits_enabled():
            # Use a temporary dataset to compute valid image IDs after filtering, then split
            temp_ds = CocoDetDataset(
                self.train_images,
                self.train_ann,
                transforms=None,
                target_category=self._target_category,
                required_instances_per_image=self._required_instances,
            )
            full_ids = list(temp_ds.img_ids)
            train_ids, val_ids, test_ids = self._compute_splits(full_ids)
            coco = temp_ds._coco  # reuse loaded COCO dict

            self.train_ds = CocoDetDataset(
                self.train_images,
                self.train_ann,
                transforms=self._train_transform,
                target_category=self._target_category,
                required_instances_per_image=self._required_instances,
                coco=coco,
                image_id_subset=train_ids,
            )
            self.val_ds = CocoDetDataset(
                self.val_images,
                self.val_ann,
                transforms=get_val_transforms(self.image_size, aspect_ratio=self._aspect_ratio),
                target_category=self._target_category,
                required_instances_per_image=self._required_instances,
                coco=coco,
                image_id_subset=val_ids,
            )
            self.test_ds = CocoDetDataset(
                self.val_images,
                self.val_ann,
                transforms=get_val_transforms(self.image_size, aspect_ratio=self._aspect_ratio),
                target_category=self._target_category,
                required_instances_per_image=self._required_instances,
                coco=coco,
                image_id_subset=test_ids,
            )
        else:
            self.train_ds = CocoDetDataset(
                self.train_images,
                self.train_ann,
                transforms=self._train_transform,
                target_category=self._target_category,
                required_instances_per_image=self._required_instances,
            )
            self.val_ds = CocoDetDataset(
                self.val_images,
                self.val_ann,
                transforms=get_val_transforms(self.image_size, aspect_ratio=self._aspect_ratio),
                target_category=self._target_category,
                required_instances_per_image=self._required_instances,
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

    def test_dataloader(self):
        if hasattr(self, "test_ds") and self.test_ds is not None:
            return DataLoader(
                self.test_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=detection_collate,
            )
        return None

    # --- Utilities for callbacks ---
    def set_resolution_mix_ratio(self, p_high: float) -> None:
        """If mixed-resolution is enabled, update the mix ratio.

        Safe to call when single-size transforms are used (no-op).
        """
        if isinstance(self._train_transform, ResolutionMixTransform):
            self._train_transform.set_p_high(p_high)

    # --- Split helpers ---
    def _splits_enabled(self) -> bool:
        cfg = self._splits_cfg or {}
        if not cfg:
            return False
        return any(k in cfg for k in ("train", "val", "test"))

    def _compute_splits(self, img_ids: Iterable[int]):
        cfg = self._splits_cfg or {}
        seed = int(cfg.get("seed", 42))
        r_train = float(cfg.get("train", 0.8))
        r_val = float(cfg.get("val", 0.2))
        r_test = float(cfg.get("test", 0.0))
        total = r_train + r_val + r_test
        if total <= 0:
            r_train, r_val, r_test = 1.0, 0.0, 0.0
            total = 1.0
        r_train, r_val, r_test = r_train / total, r_val / total, r_test / total

        ids = list(int(i) for i in img_ids)
        import random as _random

        rng = _random.Random(seed)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * r_train))
        n_val = int(round(n * r_val))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_test = max(0, n - n_train - n_val)

        train_ids = ids[:n_train]
        val_ids = ids[n_train : n_train + n_val]
        test_ids = ids[n_train + n_val :]
        return train_ids, val_ids, test_ids
