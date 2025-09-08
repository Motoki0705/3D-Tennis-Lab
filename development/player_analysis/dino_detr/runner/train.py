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

        from ..training.lit_module import DinoDetrLitModule
        from ..training.datamodule import DetectionDataModule, DataModuleConfig, AugmentationConfig
        from ..callbacks import (
            build_callbacks,
            parse_callbacks_config,
        )
        from pytorch_lightning.loggers import TensorBoardLogger

        # 1) LightningModule
        lit_module = DinoDetrLitModule(self.cfg)
        logger.info("Initialized DinoDetrLitModule")

        # 2) DataModule
        if hasattr(self.cfg, "data"):
            data_cfg = self.cfg.data
            aug_cfg = AugmentationConfig(**data_cfg.get("augmentation", {}))
            dm_cfg = DataModuleConfig(
                images_root=abspath(data_cfg.images_root),
                labeled_json=abspath(data_cfg.get("labeled_json")) if data_cfg.get("labeled_json") else None,
                train_json=abspath(data_cfg.get("train_json")) if data_cfg.get("train_json") else None,
                val_json=abspath(data_cfg.get("val_json")) if data_cfg.get("val_json") else None,
                val_ratio=float(getattr(data_cfg, "val_ratio", 0.1)),
                split_seed=int(getattr(data_cfg, "split_seed", 42)),
                img_size=tuple(data_cfg.img_size),
                batch_size=int(data_cfg.batch_size),
                num_workers=int(data_cfg.num_workers),
                augmentation=aug_cfg,
            )
            datamodule = DetectionDataModule(dm_cfg)
        else:
            raise SystemExit("cfg.data が見つかりません。")

        # 3) Trainer
        tcfg = self.cfg.training
        max_epochs = int(getattr(tcfg, "max_epochs", 30))
        accelerator = getattr(tcfg, "accelerator", "auto")
        devices = getattr(tcfg, "devices", 1)
        precision = getattr(tcfg, "precision", 32)
        grad_clip_val = float(getattr(tcfg, "grad_clip_val", 0.0))

        exp_name = getattr(self.cfg, "experiment_name", "dino_detr")
        logger_tb = TensorBoardLogger(save_dir=abspath("tb_logs"), name=exp_name)
        import os

        ckpt_dir = os.path.join(logger_tb.log_dir, "checkpoints")
        cb_cfg = parse_callbacks_config(getattr(self.cfg, "callbacks", {}))
        cb_list = build_callbacks(cb_cfg, ckpt_dir=ckpt_dir)

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            callbacks=cb_list,
            logger=logger_tb,
            log_every_n_steps=10,
            gradient_clip_val=grad_clip_val,
        )

        trainer.fit(lit_module, datamodule=datamodule)


__all__ = ["TrainRunner"]
