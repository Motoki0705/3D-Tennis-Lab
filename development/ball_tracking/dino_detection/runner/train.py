from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

try:
    import pytorch_lightning as pl
except Exception:  # pragma: no cover - defer user-friendly message
    pl = None  # type: ignore

from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig, OmegaConf

from ..callbacks import build_callbacks
from ..model import build_model
from ..training.datamodule import build_datamodule
from ..training.module import build_lit_module
from .base import BaseRunner

logger = logging.getLogger(__name__)


def _to_dict(cfg_like: Any) -> Mapping[str, Any]:
    if isinstance(cfg_like, DictConfig):
        return OmegaConf.to_container(cfg_like, resolve=True)  # type: ignore[return-value]
    if isinstance(cfg_like, Mapping):
        return cfg_like
    return {}


class TrainRunner(BaseRunner):
    def __init__(self, cfg: Any):
        super().__init__(cfg)

    def run(self):
        if pl is None:  # pragma: no cover - defensive branch
            raise SystemExit("pytorch_lightning が必要です。'pip install pytorch-lightning' を実行してください。")

        from pytorch_lightning.loggers import TensorBoardLogger

        training_cfg = _to_dict(self.cfg.get("training", {}))
        trainer_cfg = _to_dict(training_cfg.get("trainer", {}))
        lit_module_cfg = training_cfg.get("lit_module", {})

        max_epochs = int(trainer_cfg.get("max_epochs", 30))
        trainer_kwargs = {
            "accelerator": trainer_cfg.get("accelerator", "auto"),
            "devices": trainer_cfg.get("devices", 1),
            "precision": trainer_cfg.get("precision", 32),
            "gradient_clip_val": float(trainer_cfg.get("gradient_clip_val", 0.0)),
            "max_epochs": max_epochs,
            "log_every_n_steps": int(trainer_cfg.get("log_every_n_steps", 10)),
        }

        datamodule = build_datamodule(self.cfg.get("data", {}))

        model = build_model(self.cfg.get("model", {}))
        lit_module = build_lit_module(lit_module_cfg, model, max_epochs=max_epochs)
        logger.info("HeatmapLitModule ready (DINOv3 backbone + heatmap head)")

        exp_name = self.cfg.get("experiment_name", "dino_heatmap")
        tb_logger = TensorBoardLogger(save_dir=abspath("tb_logs"), name=exp_name)
        checkpoint_dir = os.path.join(tb_logger.log_dir, "checkpoints")
        callbacks = build_callbacks(self.cfg.get("callbacks", {}), checkpoint_dir=checkpoint_dir)

        trainer = pl.Trainer(callbacks=callbacks, logger=tb_logger, **trainer_kwargs)
        trainer.fit(lit_module, datamodule=datamodule)


__all__ = ["TrainRunner"]
