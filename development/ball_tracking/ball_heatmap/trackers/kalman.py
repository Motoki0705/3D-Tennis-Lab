from __future__ import annotations
import torch


class ConstAccelKF:
    def __init__(self, q_pos=1.0, q_vel=0.1, r_det=2.0):
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.r_det = r_det

    def step(self, z: torch.Tensor | None):
        # Placeholder; to be implemented with full matrices
        pass
