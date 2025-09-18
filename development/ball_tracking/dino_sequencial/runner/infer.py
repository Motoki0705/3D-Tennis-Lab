from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Optional

try:
    import pytorch_lightning as pl
except Exception:
    pl = None

from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig, OmegaConf

from ..model.net import build_model
from ..training.datamodule import build_datamodule
from ..training.module import build_lit_module
from .base import BaseRunner

logger = logging.getLogger(__name__)


def _to_dict(cfg_like: Any) -> Mapping[str, Any]:
    if isinstance(cfg_like, DictConfig):
        return OmegaConf.to_container(cfg_like, resolve=True)
    if isinstance(cfg_like, Mapping):
        return cfg_like
    return {}


def _resolve_checkpoint_path(path_like: Optional[str]) -> Optional[str]:
    if not path_like:
        return None
    if path_like == "best":
        return path_like
    return abspath(path_like)


class InferRunner(BaseRunner):
    def __init__(self, cfg: Any):
        super().__init__(cfg)

    def run(self):
        if pl is None:
            raise SystemExit("pytorch_lightning is required. Please run 'pip install pytorch-lightning'.")

        from pytorch_lightning.loggers import TensorBoardLogger

        datamodule = build_datamodule(self.cfg.get("data", {}))
        datamodule.setup("validate")

        trainer_cfg = _to_dict(self.cfg.get("training", {}).get("trainer", {}))
        lit_module_cfg = _to_dict(self.cfg.get("lit_module", {}))

        model = build_model(self.cfg.get("model", {}))
        lit_module = build_lit_module(model, lit_module_cfg)

        inference_cfg = _to_dict(self.cfg.get("inference", {}))
        ckpt_path = _resolve_checkpoint_path(inference_cfg.get("checkpoint_path"))

        exp_name = self.cfg.get("experiment_name", "dino_sequencial_heatmap")
        tb_logger = TensorBoardLogger(save_dir=abspath("tb_logs"), name=f"{exp_name}_infer")

        trainer_kwargs = {
            "accelerator": trainer_cfg.get("accelerator", "auto"),
            "devices": trainer_cfg.get("devices", 1),
            "precision": trainer_cfg.get("precision", 32),
            "logger": tb_logger,
            "callbacks": [],
        }
        trainer = pl.Trainer(**trainer_kwargs)

        trainer.validate(lit_module, datamodule=datamodule, ckpt_path=ckpt_path if ckpt_path != "best" else None)
