"""Multi-view triangulation helpers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..dataio import formats

Observation = tuple[str, formats.Detection2DEntry]


def triangulate(
    observations: Iterable[Observation],
    calibration: dict[str, formats.CalibrationEntry],
) -> formats.Vector3:
    """Triangulate a 3D point from multi-camera observations via linear DLT."""

    obs = list(observations)
    if len(obs) < 2:
        raise ValueError("At least two observations are required for triangulation")

    rows: list[np.ndarray] = []
    for camera_id, det in obs:
        calib = calibration.get(camera_id)
        if calib is None:
            raise KeyError(f"Calibration missing for camera '{camera_id}'")

        point = _detection_to_point(det)
        if point is None:
            raise ValueError(f"Detection for camera '{camera_id}' lacks a usable point")

        P = _projection_matrix(calib)
        u, v = point
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])

    A = np.vstack(rows)
    _, _, vh = np.linalg.svd(A)
    homogenous = vh[-1]
    if np.isclose(homogenous[-1], 0.0):
        raise ValueError("Triangulation resulted in a point at infinity")
    cartesian = homogenous[:3] / homogenous[3]
    return float(cartesian[0]), float(cartesian[1]), float(cartesian[2])


def _detection_to_point(det: formats.Detection2DEntry) -> tuple[float, float] | None:
    if "point" in det and det["point"] is not None:
        px, py = det["point"]
        return float(px), float(py)
    if "bbox" in det and det["bbox"] is not None:
        x1, y1, x2, y2 = map(float, det["bbox"])
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return None


def _projection_matrix(calib: formats.CalibrationEntry) -> np.ndarray:
    intrinsic = calib.get("intrinsics")
    extrinsic = calib.get("extrinsics")
    if intrinsic is None or extrinsic is None:
        raise ValueError("Calibration entry must contain 'intrinsics' and 'extrinsics'")

    fx = intrinsic.get("fx")
    fy = intrinsic.get("fy")
    cx = intrinsic.get("cx")
    cy = intrinsic.get("cy")

    if None in {fx, fy, cx, cy}:
        raise ValueError("Incomplete intrinsic parameters")

    K = np.array(
        [
            [float(fx), 0.0, float(cx)],
            [0.0, float(fy), float(cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    R = np.array(extrinsic.get("R", np.eye(3)), dtype=float)
    t = np.array(extrinsic.get("t", [0.0, 0.0, 0.0]), dtype=float).reshape(3, 1)

    Rt = np.hstack((R, t))
    return K @ Rt
