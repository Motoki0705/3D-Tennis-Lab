from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, List, Optional

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint

_DEFAULT_MONITOR = "val/loss"
_DEFAULT_MODE = "min"
_DEFAULT_SAVE_TOP_K = 3
_DEFAULT_FILENAME = "epoch={epoch}-valloss={val/loss:.4f}"
_DEFAULT_LR_INTERVAL = "epoch"
_DEFAULT_VIZ_ENABLED = False
_DEFAULT_VIZ_MAX_IMAGES = 4
_DEFAULT_VIZ_EPOCH_INTERVAL = 1


@dataclass
class CheckpointConfig:
    monitor: str = _DEFAULT_MONITOR
    mode: str = _DEFAULT_MODE
    save_top_k: int = _DEFAULT_SAVE_TOP_K
    filename: str = _DEFAULT_FILENAME


@dataclass
class CallbacksConfig:
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    lr_monitor_logging_interval: str = _DEFAULT_LR_INTERVAL
    visualization_enabled: bool = _DEFAULT_VIZ_ENABLED
    visualization_max_images: int = _DEFAULT_VIZ_MAX_IMAGES
    visualization_log_every_n_epochs: int = _DEFAULT_VIZ_EPOCH_INTERVAL


def _to_dict(cfg_like: Any) -> Mapping[str, Any]:
    if isinstance(cfg_like, DictConfig):
        return OmegaConf.to_container(cfg_like, resolve=True)  # type: ignore[return-value]
    if isinstance(cfg_like, Mapping):
        return cfg_like
    return {}


def _parse_callbacks_config(cfg_like: Any) -> CallbacksConfig:
    cfg_dict = _to_dict(cfg_like)
    checkpoint_dict = _to_dict(cfg_dict.get("checkpoint", {}))
    lr_monitor_dict = _to_dict(cfg_dict.get("lr_monitor", {}))
    visualization_dict = _to_dict(cfg_dict.get("visualization", {}))

    checkpoint_cfg = CheckpointConfig(
        monitor=str(checkpoint_dict.get("monitor", _DEFAULT_MONITOR)),
        mode=str(checkpoint_dict.get("mode", _DEFAULT_MODE)),
        save_top_k=int(checkpoint_dict.get("save_top_k", _DEFAULT_SAVE_TOP_K)),
        filename=str(checkpoint_dict.get("filename", _DEFAULT_FILENAME)),
    )

    return CallbacksConfig(
        checkpoint=checkpoint_cfg,
        lr_monitor_logging_interval=str(lr_monitor_dict.get("logging_interval", _DEFAULT_LR_INTERVAL)),
        visualization_enabled=bool(visualization_dict.get("enabled", _DEFAULT_VIZ_ENABLED)),
        visualization_max_images=int(visualization_dict.get("max_images", _DEFAULT_VIZ_MAX_IMAGES)),
        visualization_log_every_n_epochs=int(visualization_dict.get("log_every_n_epochs", _DEFAULT_VIZ_EPOCH_INTERVAL)),
    )


def build_callbacks(cfg_like: Any, *, checkpoint_dir: Optional[str] = None) -> List[Callback]:
    """Factory that converts Hydra config to Lightning callbacks."""

    cfg = _parse_callbacks_config(cfg_like)

    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        filename=cfg.checkpoint.filename,
    )

    lr_monitor = LearningRateMonitor(logging_interval=cfg.lr_monitor_logging_interval)
    callbacks: List[Callback] = [checkpoint_cb, lr_monitor]

    if cfg.visualization_enabled:
        from .visualization import HeatmapVisualizationCallback

        callbacks.append(
            HeatmapVisualizationCallback(
                max_images=cfg.visualization_max_images,
                log_every_n_epochs=cfg.visualization_log_every_n_epochs,
            )
        )

    return callbacks


__all__ = [
    "CheckpointConfig",
    "CallbacksConfig",
    "build_callbacks",
]
