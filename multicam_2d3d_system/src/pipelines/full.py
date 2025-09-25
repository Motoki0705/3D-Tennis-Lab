"""High-level orchestration pipeline for the full 2D→3D workflow."""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from . import detect2d, track2d, twoD_to_threeD

_LOGGER = logging.getLogger(__name__)


def run(cfg: DictConfig) -> None:
    """Execute detection, tracking, and 3D reconstruction in sequence."""

    _LOGGER.info("Stage 1/3: running 2D detection")
    detect2d.run(cfg)

    _LOGGER.info("Stage 2/3: running 2D tracking")
    track2d.run(cfg)

    _LOGGER.info("Stage 3/3: running 3D triangulation")
    twoD_to_threeD.run(cfg)
