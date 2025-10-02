"""Temporal smoothing utilities."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..dataio import formats


def smooth_trajectories(
    trajectories: Iterable[formats.ObjectTrajectory],
    method: str = "kalman",
    *,
    window: int = 5,
) -> list[formats.ObjectTrajectory]:
    """Apply a simple moving-average smoother to each trajectory."""

    if window < 1:
        raise ValueError("Smoothing window must be >= 1")

    if method not in {"kalman", "moving_average", "none"}:
        raise NotImplementedError(f"Temporal smoothing method '{method}' is not implemented")

    if method == "none" or window == 1:
        return [trajectory.copy() for trajectory in trajectories]

    half_window = max(1, window // 2)
    smoothed: list[formats.ObjectTrajectory] = []

    for trajectory in trajectories:
        frames = trajectory.get("frames", [])
        if not frames:
            smoothed.append({**trajectory})
            continue

        sorted_frames = sorted(frames, key=lambda fr: fr.get("timestamp", fr.get("frame_idx", 0)))
        coords = np.array([frame.get("X", (0.0, 0.0, 0.0)) for frame in sorted_frames], dtype=float)

        for idx in range(len(sorted_frames)):
            start = max(0, idx - half_window)
            end = min(len(sorted_frames), idx + half_window + 1)
            smoothed_coord = coords[start:end].mean(axis=0)
            sorted_frames[idx] = {
                **sorted_frames[idx],
                "X": tuple(float(v) for v in smoothed_coord),
            }

        smoothed.append({**trajectory, "frames": sorted_frames})

    return smoothed
