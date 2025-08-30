from __future__ import annotations

from typing import Tuple

import numpy as np
import cv2

from .camera import Intrinsics, PoseCW


def rodrigues_exp(omega: np.ndarray) -> np.ndarray:
    """so(3) vector -> rotation matrix via OpenCV Rodrigues.

    omega: (3,)
    return: (3,3) rotation matrix
    """
    R, _ = cv2.Rodrigues(np.asarray(omega, dtype=np.float64))
    return R


def pose_from_omega_t(omega: np.ndarray, t: np.ndarray) -> PoseCW:
    return PoseCW(R=rodrigues_exp(omega), t=np.asarray(t, dtype=np.float64).reshape(3))


def project_points_W(Xw: np.ndarray, K: Intrinsics, P: PoseCW) -> np.ndarray:
    """Project world points to image pixels with Brown–Conrady distortion.

    Xw: (N,3) world points (Z can be 0 for court plane)
    return: (N,2) image pixels
    """
    Xw = np.asarray(Xw, dtype=np.float64)
    Xc = (P.R @ Xw.T + P.t.reshape(3, 1)).T
    z = np.clip(Xc[:, 2], 1e-9, None)
    xn = Xc[:, :2] / z[:, None]
    x, y = xn[:, 0], xn[:, 1]
    r2 = x * x + y * y
    k1, k2, p1, p2, k3 = K.dist.k1, K.dist.k2, K.dist.p1, K.dist.p2, K.dist.k3
    radial = 1 + k1 * r2 + k2 * (r2**2) + k3 * (r2**3)
    x_d = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    y_d = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    u = K.fx * x_d + K.skew * y_d + K.cx
    v = K.fy * y_d + K.cy
    return np.stack([u, v], axis=1)


def invert_pose(P: PoseCW) -> Tuple[np.ndarray, np.ndarray]:
    Rwc = P.R.T
    twc = -Rwc @ P.t
    return Rwc, twc
