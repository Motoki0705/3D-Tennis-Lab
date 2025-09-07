from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple
from collections import deque
import numpy as np


@dataclass
class KalmanCVParams:
    # Process (acceleration) noise variance; larger -> smoother but more lag.
    process_var: float = 50.0
    # Base measurement noise variance; scaled down when confidence is high.
    measure_var: float = 25.0
    # Maximum Mahalanobis distance^2 to accept a measurement (2D chi-square ~ 9.21 ~ 99%).
    gating_chi2: float = 9.21


class BallTracker:
    """Single-object tracker for a tennis ball in image coordinates.

    - Constant-velocity Kalman filter x = [px, py, vx, vy]^T.
    - Robust gating via Mahalanobis distance.
    - Confidence-adaptive measurement noise.
    - Maintains a fixed-length trail of recent estimated positions.
    """

    def __init__(
        self,
        image_size_hw: Tuple[int, int],
        params: Optional[KalmanCVParams] = None,
        history_len: int = 30,
        min_conf: float = 0.2,
        max_missed: int = 30,
    ) -> None:
        self.H, self.W = int(image_size_hw[0]), int(image_size_hw[1])
        self.params = params or KalmanCVParams()
        self.history_len = int(history_len)
        self.min_conf = float(min_conf)
        self.max_missed = int(max_missed)

        # Kalman matrices
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        # White-noise acceleration model Q (discrete-time, dt=1)
        q = float(self.params.process_var)
        self.Q = np.array(
            [[0.25 * q, 0, 0.5 * q, 0], [0, 0.25 * q, 0, 0.5 * q], [0.5 * q, 0, q, 0], [0, 0.5 * q, 0, q]],
            dtype=np.float32,
        )

        self.reset()

    def reset(self) -> None:
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1e3  # large uncertainty until init
        self.initialized = False
        self.missed = 0
        self.trail: Deque[Tuple[float, float]] = deque(maxlen=self.history_len)

    def _clamp_xy(self, x: float, y: float) -> Tuple[float, float]:
        return float(np.clip(x, 0, self.W - 1)), float(np.clip(y, 0, self.H - 1))

    def _measurement_noise(self, conf: float) -> np.ndarray:
        # Confidence-adaptive R: higher conf -> lower noise.
        c = max(float(conf), 1e-3)
        base = float(self.params.measure_var)
        var = base / max(c, self.min_conf)
        return np.diag([var, var]).astype(np.float32)

    def predict(self) -> Tuple[float, float]:
        # x = F x; P = F P F^T + Q
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        px, py = float(self.x[0, 0]), float(self.x[1, 0])
        px, py = self._clamp_xy(px, py)
        self.x[0, 0], self.x[1, 0] = px, py
        return px, py

    def _accepts(self, z: np.ndarray, R: np.ndarray) -> bool:
        # Check Mahalanobis gating for robustness.
        y = z - (self.Hm @ self.x)
        S = self.Hm @ self.P @ self.Hm.T + R
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return True  # if ill-conditioned, fall back to accepting
        d2 = float(y.T @ Sinv @ y)
        return d2 <= self.params.gating_chi2

    def update(self, meas_xy: Optional[Tuple[float, float]], conf: float) -> Tuple[float, float]:
        # Predict step
        px, py = self.predict()

        used_measurement = False
        if meas_xy is not None and conf >= self.min_conf:
            z = np.array([[meas_xy[0]], [meas_xy[1]]], dtype=np.float32)
            R = self._measurement_noise(conf)
            if self._accepts(z, R):
                # Kalman gain K = P H^T (H P H^T + R)^-1
                S = self.Hm @ self.P @ self.Hm.T + R
                K = (self.P @ self.Hm.T) @ np.linalg.inv(S)
                # x = x + K (z - Hx)
                self.x = self.x + K @ (z - self.Hm @ self.x)
                # P = (I - K H) P
                I = np.eye(4, dtype=np.float32)
                self.P = (I - K @ self.Hm) @ self.P
                used_measurement = True

                if not self.initialized:
                    self.initialized = True
                self.missed = 0
            else:
                # Outlier: keep prediction
                self.missed += 1
        else:
            self.missed += 1

        # If not initialized and we have a measurement, hard-initialize
        if not self.initialized and meas_xy is not None and conf >= self.min_conf:
            self.x[0, 0], self.x[1, 0] = self._clamp_xy(meas_xy[0], meas_xy[1])
            self.x[2, 0], self.x[3, 0] = 0.0, 0.0
            self.P = np.eye(4, dtype=np.float32) * 100.0
            self.initialized = True
            self.missed = 0
            px, py = float(self.x[0, 0]), float(self.x[1, 0])

        # Reinitialize if lost for too long and a confident measurement appears
        if self.missed > self.max_missed:
            self.reset()

        # Save trail
        px, py = float(self.x[0, 0]), float(self.x[1, 0])
        px, py = self._clamp_xy(px, py)
        self.trail.append((px, py))
        return px, py

    def get_state(self) -> Tuple[float, float, float, float]:
        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0]), float(self.x[3, 0])

    def get_trail(self, last_n: Optional[int] = None) -> List[Tuple[float, float]]:
        if last_n is None or last_n >= len(self.trail):
            return list(self.trail)
        return list(self.trail)[-last_n:]
