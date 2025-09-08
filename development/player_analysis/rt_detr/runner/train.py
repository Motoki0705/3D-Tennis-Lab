from __future__ import annotations

import logging
from typing import Any

try:
    import pytorch_lightning as pl
except Exception:
    pl = None  # type: ignore

from hydra.utils import to_absolute_path as abspath

from .base import BaseRunner


logger = logging.getLogger(__name__)


class TrainRunner(BaseRunner):
    def __init__(self, cfg: Any):
        super().__init__(cfg)

    def run(self):
        if pl is None:
            raise SystemExit("pytorch_lightning が必要です。'pip install pytorch-lightning' を実行してください。")

        from ..training.lit_module import RTDetrLightning
        from ..training.datamodule import PlayerDataModule, DataModuleConfig, AugmentationConfig
        from pytorch_lightning.loggers import TensorBoardLogger
        from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

        # 1) LightningModule
        lit_module = RTDetrLightning(self.cfg)
        logger.info("Initialized RTDETRLitModule")

        # 2) DataModule（cfg.data）
        datamodule = None
        if hasattr(self.cfg, "data"):
            data_cfg = self.cfg.data
            aug_cfg = AugmentationConfig(**data_cfg.get("augmentation", {}))
            dm_cfg = DataModuleConfig(
                images_root=abspath(data_cfg.images_root),
                labeled_json=abspath(data_cfg.labeled_json),
                img_size=tuple(data_cfg.img_size),
                batch_size=int(data_cfg.batch_size),
                num_workers=int(data_cfg.num_workers),
                val_ratio=float(getattr(data_cfg, "val_ratio", 0.1)),
                split_seed=int(getattr(data_cfg, "split_seed", 42)),
                category_name=str(getattr(data_cfg, "category_name", "player")),
                augmentation=aug_cfg,
            )
            datamodule = PlayerDataModule(dm_cfg)
        else:
            logger.warning("cfg.data が見つからないため、DataModuleは作成しません。")

        # 3) Trainer
        tcfg = self.cfg.training
        max_epochs = int(getattr(tcfg, "max_epochs", 50))
        accelerator = getattr(tcfg, "accelerator", "auto")
        devices = getattr(tcfg, "devices", 1)
        precision = getattr(tcfg, "precision", 16)

        exp_name = getattr(self.cfg, "experiment_name", "rt_detr_finetune")
        logger_tb = TensorBoardLogger(save_dir=abspath("tb_logs"), name=exp_name)

        import os

        ckpt_dir = os.path.join(logger_tb.log_dir, "checkpoints")
        callbacks = [
            ModelCheckpoint(
                dirpath=ckpt_dir,
                monitor="val/loss",
                mode="min",
                save_top_k=3,
                filename="epoch={epoch}-valloss={val/loss:.4f}",
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            callbacks=callbacks,
            logger=logger_tb,
            log_every_n_steps=10,
        )

        trainer.fit(lit_module, datamodule=datamodule)


__all__ = ["TrainRunner"]
