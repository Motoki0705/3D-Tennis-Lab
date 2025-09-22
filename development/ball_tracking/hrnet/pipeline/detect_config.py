from __future__ import annotations

from pathlib import Path
from typing import Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def load_detect_config(overrides: Sequence[str] | None = None) -> DictConfig:
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    overrides = list(overrides or [])
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="detect", overrides=list(overrides))
    return cfg
