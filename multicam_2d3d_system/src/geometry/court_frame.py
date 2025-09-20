"""Court coordinate frame utilities."""

from __future__ import annotations

from ..dataio import formats


def to_court_frame(
    reconstruction: formats.Reconstruction3D,
    calibration: dict[str, formats.CalibrationEntry],
) -> formats.Reconstruction3D:
    """Transform reconstruction into the standardized court frame."""
    raise NotImplementedError("Implement court frame transforms")
