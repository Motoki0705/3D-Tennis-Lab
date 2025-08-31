from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class KFState:
    x: np.ndarray  # state vector [x, y, vx, vy]
    P: np.ndarray  # covariance


class ConstantVelocityKF:
    def __init__(self, dt: float = 1.0, process_var: float = 1.0, meas_var: float = 1.0):
        self.dt = dt
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.Q = process_var * np.eye(4)
        self.R = meas_var * np.eye(2)

    def init(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0, pos_var: float = 1.0) -> KFState:
        P = np.diag([pos_var, pos_var, 10.0, 10.0])
        return KFState(x=np.array([x, y, vx, vy], dtype=float), P=P)

    def predict(self, state: KFState) -> KFState:
        x_pred = self.F @ state.x
        P_pred = self.F @ state.P @ self.F.T + self.Q
        return KFState(x=x_pred, P=P_pred)

    def update(self, state: KFState, z: Optional[np.ndarray], meas_var_scale: float = 1.0) -> KFState:
        if z is None:
            return state  # no measurement
        R = self.R * meas_var_scale
        y = z - (self.H @ state.x)
        S = self.H @ state.P @ self.H.T + R
        K = state.P @ self.H.T @ np.linalg.inv(S)
        x_new = state.x + K @ y
        P_new = (np.eye(4) - K @ self.H) @ state.P
        return KFState(x=x_new, P=P_new)
