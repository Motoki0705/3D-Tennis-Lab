"""Multi-view triangulation helpers."""

from __future__ import annotations

from collections.abc import Iterable

from ..dataio import formats

Observation = tuple[str, formats.Detection2DEntry]


def triangulate(
    observations: Iterable[Observation],
    calibration: dict[str, formats.CalibrationEntry],
) -> formats.Vector3:
    """Triangulate a 3D point from multi-camera observations."""
    raise NotImplementedError("Implement triangulation logic")
