from __future__ import annotations

import logging
import os
from typing import Any

import pytorch_lightning as pl
from hydra.utils import to_absolute_path as abspath
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .base_runner import BaseRunner
from ..training.lit_module import SequenceLitModule
from ..training.datamodule import BallDataModule, DataModuleConfig
from ..utils.normalization import compute_feature_stats

logger = logging.getLogger(__name__)


class TrainRunner(BaseRunner):
    def __init__(self, cfg: Any):
        super().__init__(cfg)

    def run(self):
        pl.seed_everything(self.cfg.data.split_seed, workers=True)

        # 1) DataModule
        dm_cfg = DataModuleConfig(
            labeled_json=abspath(self.cfg.data.labeled_json),
            sequence_length=int(self.cfg.data.sequence_length),
            predict_offset=int(self.cfg.data.predict_offset),
            val_ratio=float(self.cfg.data.val_ratio),
            split_seed=int(self.cfg.data.split_seed),
            batch_size=int(self.cfg.data.batch_size),
            num_workers=int(self.cfg.data.num_workers),
        )
        datamodule = BallDataModule(dm_cfg)
        datamodule.setup("fit")
        logger.info("Initialized BallDataModule and prepared datasets")

        # 2) Compute normalization stats from train samples
        stats = compute_feature_stats(datamodule.train_dataset.samples)  # type: ignore

        # 3) LightningModule with stats
        lit_module = SequenceLitModule(self.cfg, feature_stats=stats)
        logger.info("Initialized SequenceLitModule")

        # 3) Callbacks & Logger
        exp_name = self.cfg.get("experiment_name", "ball_tracker_exp")
        logger_tb = TensorBoardLogger(save_dir=abspath("tb_logs"), name=exp_name)

        ckpt_dir = os.path.join(logger_tb.log_dir, "checkpoints")
        cb_cfg = self.cfg.get("callbacks", {})
        checkpoint_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor=cb_cfg.checkpoint.monitor,
            mode=cb_cfg.checkpoint.mode,
            save_top_k=cb_cfg.checkpoint.save_top_k,
            filename=cb_cfg.checkpoint.filename,
        )
        lr_monitor_cb = LearningRateMonitor(logging_interval=cb_cfg.lr_monitor.logging_interval)
        callbacks = [checkpoint_cb, lr_monitor_cb]
        es_cfg = getattr(cb_cfg, "early_stopping", None)
        if es_cfg is not None:
            callbacks.append(
                EarlyStopping(
                    monitor=es_cfg.monitor,
                    mode=es_cfg.mode,
                    patience=int(es_cfg.patience),
                )
            )

        # 4) Trainer
        tcfg = self.cfg.training
        trainer = pl.Trainer(
            max_epochs=int(tcfg.max_epochs),
            accelerator=tcfg.accelerator,
            devices=tcfg.devices,
            precision=tcfg.precision,
            callbacks=callbacks,
            logger=logger_tb,
            log_every_n_steps=10,
            gradient_clip_val=float(getattr(tcfg, "gradient_clip_val", 0.0)),
        )

        logger.info("Starting training...")
        trainer.fit(lit_module, datamodule=datamodule)
        logger.info("Training finished.")


# Add a BaseRunner to runner folder to be compliant with the architecture
