from __future__ import annotations


import torch

from .base import TemporalBlock


class TemporalShift(TemporalBlock):
    """Temporal Shift Module (near-zero params/compute)."""

    def __init__(self, channels: int, shift_ratio: float = 1 / 8):
        super().__init__()
        self.fold = max(1, int(channels * float(shift_ratio)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        f = self.fold
        if f == 0 or T == 1:
            return x
        out = torch.empty_like(x)
        # back shift (uses previous time)
        out[:, 1:, :f] = x[:, :-1, :f]
        out[:, :1, :f] = 0
        # forward shift
        out[:, :-1, f : 2 * f] = x[:, 1:, f : 2 * f]
        out[:, -1:, f : 2 * f] = 0
        # keep
        out[:, :, 2 * f :] = x[:, :, 2 * f :]
        return out
