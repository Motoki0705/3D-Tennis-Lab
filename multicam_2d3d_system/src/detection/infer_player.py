"""Player detection inference wrapper."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..dataio import formats


def run_player_inference(
    video_paths: Iterable[Path],
    weights_path: Path,
) -> list[formats.FrameDetections2D]:
    """Run player detector on the provided videos and return detection records."""
    raise NotImplementedError("Implement player detector integration")
