from __future__ import annotations

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from omegaconf import DictConfig

from development.ball_tracking.ball_heatmap.training.dataset import BallDataset, DataConfig
from development.ball_tracking.ball_heatmap.training.collate import collate_batch


class BallDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.dc = DataConfig(
            images_root=cfg.data.images_root,
            labeled_json=cfg.data.labeled_json,
            unlabeled_json=cfg.data.unlabeled_json,
            img_size=tuple(cfg.data.img_size),
            T=int(cfg.data.T),
            frame_stride=int(cfg.data.frame_stride),
            scales=list(cfg.data.scales),
            sigma_px=list(cfg.data.sigma_px),
            supervise_hm_on_v1=bool(cfg.data.get("supervise_hm_on_v1", False)),
        )
        self.batch_size = cfg.data.get("batch_size", 4)
        self.num_workers = cfg.data.get("num_workers", 4)
        self.pin_memory = cfg.data.get("pin_memory", True)
        self.drop_last = cfg.data.get("drop_last", True)
        self.shuffle = cfg.data.get("shuffle", True)
        self.val_split = cfg.data.get("val_split", 0.1)
        self.persistent_workers = cfg.data.get("persistent_workers", True)

    def setup(self, stage: str | None = None) -> None:
        # Labeled dataset
        full_labeled = BallDataset(self.dc, labeled=True, semisup=False)
        n_total = len(full_labeled)
        n_val = int(self.val_split * n_total)
        n_train = n_total - n_val
        self.ds_labeled_train, self.ds_labeled_val = random_split(full_labeled, [n_train, n_val])
        # Unlabeled dataset
        self.ds_unlabeled = BallDataset(self.dc, labeled=False, semisup=self.cfg.semisup.get("enable", False))

    def train_dataloader(self):
        # For initial scaffold, return labeled loader. Unlabeled loader is available as attribute.
        self.unlabeled_loader = None
        if self.cfg.semisup.get("enable", False):
            self.unlabeled_loader = DataLoader(
                self.ds_unlabeled,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
                collate_fn=collate_batch,
                persistent_workers=self.persistent_workers,
            )
        return DataLoader(
            self.ds_labeled_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=collate_batch,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_labeled_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=collate_batch,
            persistent_workers=self.persistent_workers,
        )
