from __future__ import annotations

import logging
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar, EarlyStopping

from .base import BaseRunner


logger = logging.getLogger(__name__)


class TrainRunner(BaseRunner):
    def __init__(self, cfg: Any):
        super().__init__(cfg)

    def run(self):
        from ..training.lit_module import LiteUNetContextLitModule
        from ..training.datamodule import CourtDataModule, DataConfig, AugmentationConfig

        # LightningModule
        lit_module = LiteUNetContextLitModule(self.cfg)
        logger.info("Initialized LiteUNetContextLitModule")

        # DataModule
        dcfg_raw = getattr(self.cfg, "data", None)
        if dcfg_raw is None:
            raise SystemExit("cfg.data が見つかりません。configs/data を確認してください。")

        aug_cfg = AugmentationConfig(**getattr(dcfg_raw, "augmentation", {}))
        dcfg = DataConfig(
            img_dir=str(dcfg_raw.img_dir),
            annotation_file=str(dcfg_raw.annotation_file),
            img_size=tuple(dcfg_raw.img_size),
            output_stride=int(getattr(dcfg_raw, "output_stride", getattr(self.cfg.model, "out_stride", 4))),
            sigma=float(getattr(dcfg_raw, "sigma", 2.0)),
            batch_size=int(dcfg_raw.batch_size),
            num_workers=int(dcfg_raw.num_workers),
            pin_memory=bool(getattr(dcfg_raw, "pin_memory", True)),
            persistent_workers=bool(getattr(dcfg_raw, "persistent_workers", True)),
            train_ratio=float(getattr(dcfg_raw, "train_ratio", 0.8)),
            val_ratio=float(getattr(dcfg_raw, "val_ratio", 0.1)),
            use_offset=bool(getattr(dcfg_raw, "use_offset", getattr(self.cfg.model, "use_offset_head", False))),
            augmentation=aug_cfg,
        )
        dm = CourtDataModule(
            dcfg,
            deep_supervision=bool(getattr(self.cfg.model, "deep_supervision", False)),
            num_keypoints=int(getattr(self.cfg.model, "num_keypoints", 15)),
        )

        # Trainer
        tcfg = getattr(self.cfg, "training", {})
        exp_name = getattr(self.cfg, "experiment_name", "lite_unet_context")
        logger_tb = TensorBoardLogger(save_dir="tb_logs", name=exp_name)

        ckpt = ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            filename="epoch={epoch}-valloss={val/loss:.4f}",
            auto_insert_metric_name=False,
        )
        lrmon = LearningRateMonitor(logging_interval="epoch")
        early = EarlyStopping(monitor="val/loss", mode="min", patience=int(getattr(tcfg, "early_stop_patience", 10)))
        progress = RichProgressBar()
        # Heatmap visualizer
        try:
            from ....utils.callbacks.heatmap_logger import HeatmapImageLogger

            cb_cfg = getattr(self.cfg, "callbacks", None)
            hm_cfg = getattr(cb_cfg, "heatmap_logger", None) if cb_cfg is not None else None
            num_samples = int(getattr(hm_cfg, "num_samples", 3)) if hm_cfg is not None else 3
            heatmap_cb = HeatmapImageLogger(num_samples=num_samples)
        except Exception:
            heatmap_cb = None

        callbacks = [ckpt, lrmon, early, progress]
        if heatmap_cb is not None:
            callbacks.append(heatmap_cb)

        trainer = pl.Trainer(
            max_epochs=int(getattr(tcfg, "max_epochs", 50)),
            accelerator=getattr(tcfg, "accelerator", "auto"),
            devices=getattr(tcfg, "devices", 1),
            precision=getattr(tcfg, "precision", 16),
            callbacks=callbacks,
            logger=logger_tb,
            log_every_n_steps=10,
        )

        trainer.fit(lit_module, datamodule=dm)


__all__ = ["TrainRunner"]
