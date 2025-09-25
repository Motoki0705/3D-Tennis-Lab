"""Court coordinate frame utilities."""

from __future__ import annotations

import logging

import numpy as np

from ..dataio import formats

_LOGGER = logging.getLogger(__name__)


def to_court_frame(
    reconstruction: formats.Reconstruction3D,
    calibration: dict[str, formats.CalibrationEntry],
) -> formats.Reconstruction3D:
    """Transform reconstruction into a canonical court frame.

    The current implementation normalises the reconstruction by translating the
    origin to the average camera position. This is a pragmatic fallback until a
    full court registration pipeline is available.
    """

    if not reconstruction or "objects" not in reconstruction:
        return reconstruction

    camera_positions = _camera_world_positions(calibration)
    if not camera_positions:
        _LOGGER.info("Court frame transformation skipped (no calibration positions available)")
        return reconstruction

    origin = np.mean(camera_positions, axis=0)

    transformed_objects: list[formats.ObjectTrajectory] = []
    for obj in reconstruction.get("objects", []):
        frames = []
        for frame in obj.get("frames", []):
            coord = frame.get("X")
            if coord is None:
                frames.append(frame)
                continue
            transformed = np.array(coord, dtype=float) - origin
            frames.append({**frame, "X": tuple(float(v) for v in transformed)})
        transformed_objects.append({**obj, "frames": frames})

    output: formats.Reconstruction3D = {
        **reconstruction,
        "objects": transformed_objects,
        "court_frame": {"origin": "camera-centroid", "axes": "approx"},
    }
    return output


def _camera_world_positions(calibration: dict[str, formats.CalibrationEntry]) -> list[np.ndarray]:
    positions: list[np.ndarray] = []
    for entry in calibration.values():
        extrinsic = entry.get("extrinsics")
        if extrinsic is None:
            continue
        R = np.array(extrinsic.get("R", np.eye(3)), dtype=float)
        t = np.array(extrinsic.get("t", [0.0, 0.0, 0.0]), dtype=float)
        try:
            cam_pos = -R.T @ t
        except Exception:  # pragma: no cover - safety net
            continue
        positions.append(cam_pos)
    return positions
