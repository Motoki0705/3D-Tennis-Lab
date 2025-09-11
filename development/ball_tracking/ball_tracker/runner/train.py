from __future__ import annotations

import logging
import os
from typing import Any

import pytorch_lightning as pl
import hydra
from hydra.utils import to_absolute_path as abspath
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
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

        # Instantiate callbacks
        callbacks: list[pl.Callback] = []
        cb_cfg = self.cfg.get("callbacks")
        if cb_cfg:
            # Handle callbacks that need special instantiation or have no _target_
            if "checkpoint" in cb_cfg:
                ckpt_dir = os.path.join(logger_tb.log_dir, "checkpoints")
                callbacks.append(ModelCheckpoint(dirpath=ckpt_dir, filename=cb_cfg.checkpoint.filename))

            if "early_stopping" in cb_cfg:
                callbacks.append(EarlyStopping(**cb_cfg.early_stopping))

            # Instantiate all other callbacks from config via hydra
            for name, cb_conf in cb_cfg.items():
                if name not in ["checkpoint", "early_stopping"] and "_target_" in cb_conf:
                    callbacks.append(hydra.utils.instantiate(cb_conf))

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
