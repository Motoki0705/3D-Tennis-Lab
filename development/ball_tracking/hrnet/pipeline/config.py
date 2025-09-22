from __future__ import annotations

from pathlib import Path
from typing import Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def load_config(overrides: Sequence[str] | None = None) -> DictConfig:
    """Load Hydra configuration from the local conf directory."""

    overrides = list(overrides or [])
    conf_dir = Path(__file__).with_suffix("").parent / "conf"

    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    return cfg
