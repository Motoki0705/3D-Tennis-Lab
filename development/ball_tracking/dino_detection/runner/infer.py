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


class InferRunner(BaseRunner):
    def __init__(self, cfg: Any):
        super().__init__(cfg)

    def run(self):
        if pl is None:
            raise SystemExit("pytorch_lightning が必要です。'pip install pytorch-lightning' を実行してください。")

        from ..training.module import HeatmapLitModule
        from ..training.datamodule import BallHeatmapDataModule, DataModuleConfig
        from pytorch_lightning.loggers import TensorBoardLogger
        import os

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
        datamodule.setup("validate")

        # Build module and optionally load checkpoint
        lit_module = HeatmapLitModule(self.cfg)
        ckpt_path = self.cfg.get("inference", {}).get("checkpoint_path", None)

        exp_name = self.cfg.get("experiment_name", "dino_heatmap")
        logger_tb = TensorBoardLogger(save_dir=abspath("tb_logs"), name=f"{exp_name}_infer")
        ckpt_dir = os.path.join(logger_tb.log_dir, "checkpoints")
        callbacks = []  # no callbacks necessary for simple inference

        trainer = pl.Trainer(
            accelerator=self.cfg.get("training", {}).get("accelerator", "auto"),
            devices=self.cfg.get("training", {}).get("devices", 1),
            logger=logger_tb,
            callbacks=callbacks,
        )

        if ckpt_path and ckpt_path != "best":
            trainer.validate(lit_module, datamodule=datamodule, ckpt_path=abspath(ckpt_path))
        else:
            trainer.validate(lit_module, datamodule=datamodule)


__all__ = ["InferRunner"]
