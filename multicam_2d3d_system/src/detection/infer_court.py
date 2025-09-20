"""Court layout detection wrapper."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..dataio import formats


def run_court_inference(
    image_paths: Iterable[Path],
    weights_path: Path,
) -> list[formats.FrameDetections2D]:
    """Detect court key points or segmentation masks."""
    raise NotImplementedError("Implement court detector integration")
