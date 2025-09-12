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


def _build_callbacks(cb_cfg_like: Any, ckpt_dir: str | None):
    try:
        from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pytorch_lightning is required to build callbacks") from e

    # Simple parser from dict-like config
    monitor = cb_cfg_like.get("checkpoint", {}).get("monitor", "val/loss")
    mode = cb_cfg_like.get("checkpoint", {}).get("mode", "min")
    save_top_k = int(cb_cfg_like.get("checkpoint", {}).get("save_top_k", 3))
    filename = cb_cfg_like.get("checkpoint", {}).get("filename", "epoch={epoch}-valloss={val/loss:.4f}")

    if ckpt_dir is not None:
        checkpoint_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            filename=filename,
        )
    else:
        checkpoint_cb = ModelCheckpoint(
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            filename=filename,
        )

    lr_interval = cb_cfg_like.get("lr_monitor", {}).get("logging_interval", "epoch")
    lr_monitor = LearningRateMonitor(logging_interval=lr_interval)
    return [checkpoint_cb, lr_monitor]


class TrainRunner(BaseRunner):
    def __init__(self, cfg: Any):
        super().__init__(cfg)

    def run(self):
        if pl is None:
            raise SystemExit("pytorch_lightning が必要です。'pip install pytorch-lightning' を実行してください。")

        from ..training.module import HeatmapLitModule
        from ..training.datamodule import BallHeatmapDataModule, DataModuleConfig
        from pytorch_lightning.loggers import TensorBoardLogger
        import os

        # 1) LightningModule
        lit_module = HeatmapLitModule(self.cfg)
        logger.info("Initialized HeatmapLitModule (DINOv3+FPN+HeatmapHead)")

        # 2) DataModule
        data_cfg = self.cfg.get("data", {})
        dm_cfg = DataModuleConfig(
            images_root=abspath(data_cfg.get("images_root", "data/images")),
            labeled_json=abspath(data_cfg.get("labeled_json", "data/annotations.json")),
            img_size=tuple(data_cfg.get("img_size", [640, 640])),
            output_stride=int(data_cfg.get("output_stride", 4)),
            sigma_px=float(data_cfg.get("sigma_px", 2.0)),
            batch_size=int(data_cfg.get("batch_size", 8)),
            num_workers=int(data_cfg.get("num_workers", 4)),
            val_ratio=float(data_cfg.get("val_ratio", 0.1)),
            seed=int(data_cfg.get("split_seed", 42)),
        )
        datamodule = BallHeatmapDataModule(dm_cfg)

        # 3) Trainer
        tcfg = self.cfg.get("training", {})
        max_epochs = int(tcfg.get("max_epochs", 30))
        accelerator = tcfg.get("accelerator", "auto")
        devices = tcfg.get("devices", 1)
        precision = tcfg.get("precision", 32)
        grad_clip = float(tcfg.get("gradient_clip_val", 0.0))

        exp_name = self.cfg.get("experiment_name", "dino_heatmap")
        logger_tb = TensorBoardLogger(save_dir=abspath("tb_logs"), name=exp_name)
        ckpt_dir = os.path.join(logger_tb.log_dir, "checkpoints")
        callbacks = _build_callbacks(self.cfg.get("callbacks", {}), ckpt_dir)

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            gradient_clip_val=grad_clip,
            callbacks=callbacks,
            logger=logger_tb,
            log_every_n_steps=10,
        )

        trainer.fit(lit_module, datamodule=datamodule)


__all__ = ["TrainRunner"]
