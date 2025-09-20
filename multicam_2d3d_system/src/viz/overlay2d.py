"""2D overlay visualization helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..dataio import formats


def render_overlays(
    frames: Iterable[formats.FrameDetections2D],
    output_dir: Path,
) -> None:
    """Render bounding boxes and tracks onto video frames."""
    raise NotImplementedError("Implement 2D overlay rendering")
