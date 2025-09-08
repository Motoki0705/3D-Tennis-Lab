from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from .tb_image_logger import TensorBoardDetectionImageLogger


@dataclass
class TBImageLoggerConfig:
    every_n_steps: int = 200
    num_samples: int = 2
    mode: str = "val"  # "train" or "val"


@dataclass
class CheckpointConfig:
    monitor: str = "val/loss"
    mode: str = "min"
    save_top_k: int = 3
    filename: str = "epoch={epoch}-valloss={val/loss:.4f}"


@dataclass
class LRMonitorConfig:
    logging_interval: str = "epoch"


@dataclass
class CallbacksConfig:
    tb_image_logger: Optional[TBImageLoggerConfig] = None
    checkpoint: Optional[CheckpointConfig] = None
    lr_monitor: Optional[LRMonitorConfig] = None


def _to_plain_dict(cfg: Any) -> dict:
    try:
        from omegaconf import DictConfig, OmegaConf  # type: ignore

        if isinstance(cfg, DictConfig):
            return dict(OmegaConf.to_container(cfg, resolve=True))
    except Exception:
        pass
    return dict(cfg) if isinstance(cfg, dict) else {}


def parse_callbacks_config(cfg_like: Any) -> CallbacksConfig:
    d = _to_plain_dict(cfg_like)

    tb = d.get("tb_image_logger")
    cp = d.get("checkpoint")
    lr = d.get("lr_monitor")

    tb_dc = TBImageLoggerConfig(**tb) if isinstance(tb, dict) else None
    cp_dc = CheckpointConfig(**cp) if isinstance(cp, dict) else None
    lr_dc = LRMonitorConfig(**lr) if isinstance(lr, dict) else None

    return CallbacksConfig(tb_image_logger=tb_dc, checkpoint=cp_dc, lr_monitor=lr_dc)


def build_callbacks(cfg: CallbacksConfig, ckpt_dir: Optional[str] = None) -> List[Any]:
    try:
        import pytorch_lightning as pl  # noqa: F401
        from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    except Exception as e:
        raise RuntimeError("pytorch_lightning is required to build callbacks") from e

    callbacks: List[Any] = []

    cp_cfg = cfg.checkpoint or CheckpointConfig()
    if ckpt_dir is not None:
        checkpoint_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor=cp_cfg.monitor,
            mode=cp_cfg.mode,
            save_top_k=cp_cfg.save_top_k,
            filename=cp_cfg.filename,
        )
    else:
        checkpoint_cb = ModelCheckpoint(
            monitor=cp_cfg.monitor,
            mode=cp_cfg.mode,
            save_top_k=cp_cfg.save_top_k,
            filename=cp_cfg.filename,
        )
    callbacks.append(checkpoint_cb)

    lr_cfg = cfg.lr_monitor or LRMonitorConfig()
    callbacks.append(LearningRateMonitor(logging_interval=lr_cfg.logging_interval))

    if cfg.tb_image_logger is not None:
        t = cfg.tb_image_logger
        callbacks.append(
            TensorBoardDetectionImageLogger(
                every_n_steps=int(t.every_n_steps),
                num_samples=int(t.num_samples),
                mode=str(t.mode),
            )
        )

    return callbacks


__all__ = [
    "TensorBoardDetectionImageLogger",
    "TBImageLoggerConfig",
    "CheckpointConfig",
    "LRMonitorConfig",
    "CallbacksConfig",
    "parse_callbacks_config",
    "build_callbacks",
]
