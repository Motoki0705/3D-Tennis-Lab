"""Multi-object tracking utilities."""

from __future__ import annotations

from collections.abc import Iterable

from ..dataio import formats


def track_detections(
    detections: Iterable[formats.FrameDetections2D],
    method: str = "bytetrack",
) -> list[formats.CameraTracks]:
    """Assign track identifiers to per-frame detections."""
    raise NotImplementedError("Implement tracking wrapper")
