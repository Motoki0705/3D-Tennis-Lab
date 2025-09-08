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
        from ..training.lit_module import CourtLitModule
        from ..training.datamodule import CourtDataModule
        from development.utils.transformers.keypoint_transformer import prepare_transforms

        # LightningModule
        lit_module = CourtLitModule(self.cfg)
        logger.info("Initialized CourtLitModule (ViT Heatmap)")

        # DataModule with transforms
        train_tf, val_tf = prepare_transforms(img_size=self.cfg.dataset.img_size)
        datamodule = CourtDataModule(
            config=self.cfg,
            train_transforms=train_tf,
            val_transforms=val_tf,
            test_transforms=val_tf,
        )

        # Trainer
        tcfg = self.cfg.training
        exp_name = getattr(self.cfg, "experiment_name", "vit_heatmap")
        logger_tb = TensorBoardLogger(save_dir="tb_logs", name=exp_name)

        ckpt = ModelCheckpoint(
            monitor=self.cfg.callbacks.checkpoint.monitor,
            mode=self.cfg.callbacks.checkpoint.mode,
            save_top_k=int(self.cfg.callbacks.checkpoint.save_top_k),
            filename="epoch={epoch}-pck={val/PCK@0.05:.4f}",
            auto_insert_metric_name=False,
        )
        lrmon = LearningRateMonitor(logging_interval="epoch")
        early = EarlyStopping(
            monitor=self.cfg.callbacks.early_stopping.monitor,
            mode=self.cfg.callbacks.early_stopping.mode,
            patience=int(self.cfg.callbacks.early_stopping.patience),
        )
        progress = RichProgressBar()

        # Heatmap visualizer
        try:
            from development.utils.callbacks.heatmap_logger import HeatmapImageLogger

            num_samples = int(getattr(self.cfg.callbacks.heatmap_logger, "num_samples", 3))
            heatmap_cb = HeatmapImageLogger(num_samples=num_samples)
        except Exception:
            heatmap_cb = None

        callbacks = [ckpt, lrmon, early, progress]
        if heatmap_cb is not None:
            callbacks.append(heatmap_cb)

        trainer = pl.Trainer(
            max_epochs=int(tcfg.max_epochs),
            accelerator=getattr(tcfg, "accelerator", "auto"),
            devices=getattr(tcfg, "devices", 1),
            precision=getattr(tcfg, "precision", 16),
            callbacks=callbacks,
            logger=logger_tb,
        )

        trainer.fit(lit_module, datamodule=datamodule)


__all__ = ["TrainRunner"]
