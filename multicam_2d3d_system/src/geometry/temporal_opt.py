"""Temporal smoothing utilities."""

from __future__ import annotations

from collections.abc import Iterable

from ..dataio import formats


def smooth_trajectories(
    trajectories: Iterable[formats.ObjectTrajectory],
    method: str = "kalman",
) -> list[formats.ObjectTrajectory]:
    """Apply temporal smoothing to 3D trajectories."""
    raise NotImplementedError("Implement temporal optimization")
