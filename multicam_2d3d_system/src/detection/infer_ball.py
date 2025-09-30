"""Ball detection inference wrapper."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from omegaconf import DictConfig

from ..dataio import formats


def run_ball_inference(
    video_paths: Iterable[Path],
    cfg: DictConfig,
) -> list[formats.FrameDetections2D]:
    """Run ball detector on the provided videos and return detection records."""
    raise NotImplementedError("Implement ball detector integration")
