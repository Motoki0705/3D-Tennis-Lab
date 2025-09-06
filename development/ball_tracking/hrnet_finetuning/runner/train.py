from __future__ import annotations

import logging
from typing import Any

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
except Exception as e:
    # Lazy import error will be raised at runtime in run()
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

        from ..training.lit_module import HRNetFinetuneLitModule
        from ..training.datamodule import BallDataModule, DataModuleConfig, AugmentationConfig
        from ..callbacks import TensorBoardHeatmapLogger
        from pytorch_lightning.loggers import TensorBoardLogger

        # 1) LightningModule（HRNet3DStem内でpretrained_checkpointもロード）
        lit_module = HRNetFinetuneLitModule(self.cfg)
        logger.info("Initialized HRNetFinetuneLitModule")

        # 2) DataModule（cfg.data があれば）
        datamodule = None
        if hasattr(self.cfg, "data"):
            data_cfg = self.cfg.data
            aug_cfg = AugmentationConfig(**data_cfg.get("augmentation", {}))
            dm_cfg = DataModuleConfig(
                images_root=abspath(data_cfg.images_root),
                labeled_json_train=abspath(data_cfg.labeled_json_train),
                labeled_json_val=abspath(data_cfg.labeled_json_val),
                img_size=tuple(data_cfg.img_size),
                T=int(data_cfg.T),
                frame_stride=int(data_cfg.frame_stride),
                output_stride=int(data_cfg.output_stride),
                sigma_px=float(data_cfg.sigma_px),
                batch_size=int(data_cfg.batch_size),
                num_workers=int(data_cfg.num_workers),
                augmentation=aug_cfg,
            )
            datamodule = BallDataModule(dm_cfg)
        else:
            logger.warning("cfg.data が見つからないため、DataModuleは作成しません。")

        # 3) Trainer
        tcfg = self.cfg.training
        max_epochs = int(getattr(tcfg, "max_epochs", 30))
        accelerator = getattr(tcfg, "accelerator", "auto")
        devices = getattr(tcfg, "devices", 1)
        precision = getattr(tcfg, "precision", 32)

        lr_cb = LearningRateMonitor(logging_interval="epoch")
        es_cb = EarlyStopping(monitor="val/loss", mode="min", patience=5)

        # TensorBoard logger under tb_logs/<experiment_name>/version_*/
        exp_name = getattr(self.cfg, "experiment_name", "hrnet_finetune")
        logger_tb = TensorBoardLogger(save_dir=abspath("tb_logs"), name=exp_name)
        # Checkpoint inside version dir
        import os

        ckpt_dir = os.path.join(logger_tb.log_dir, "checkpoints")
        ckpt_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            filename="epoch={epoch}-valloss={val/loss:.4f}",
        )

        # Optional heatmap callback
        cb_list = [ckpt_cb, lr_cb, es_cb]
        if hasattr(self.cfg, "callbacks") and hasattr(self.cfg.callbacks, "tb_image_logger"):
            c = self.cfg.callbacks.tb_image_logger
            cb_list.append(
                TensorBoardHeatmapLogger(
                    every_n_steps=int(getattr(c, "every_n_steps", 200)),
                    num_samples=int(getattr(c, "num_samples", 2)),
                    mode=str(getattr(c, "mode", "val")),
                    log_overlay=bool(getattr(c, "log_overlay", True)),
                )
            )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            callbacks=cb_list,
            logger=logger_tb,
            log_every_n_steps=10,
        )

        trainer.fit(lit_module, datamodule=datamodule)


__all__ = ["TrainRunner"]
