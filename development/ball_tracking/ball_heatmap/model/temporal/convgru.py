from __future__ import annotations


import torch
import torch.nn as nn

from .base import TemporalBlock


class ConvGRUCell(nn.Module):
    def __init__(self, channels: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.gates = nn.Conv2d(channels * 2, channels * 2, k, padding=p, bias=True)  # z, r
        self.cand = nn.Conv2d(channels * 2, channels, k, padding=p, bias=True)

    def forward(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        xh = torch.cat([x_t, h], dim=1)
        z, r = self.gates(xh).chunk(2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)
        n = torch.tanh(self.cand(torch.cat([x_t, r * h], dim=1)))
        return (1 - z) * n + z * h


class TemporalConvGRU(TemporalBlock):
    """Causal ConvGRU over time."""

    def __init__(self, channels: int, kernel_t: int = 3):
        super().__init__()
        self.cell = ConvGRUCell(channels, kernel_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        h = x.new_zeros(B, C, H, W)
        outs = []
        for t in range(T):
            h = self.cell(x[:, t], h)
            outs.append(h)
        return torch.stack(outs, dim=1)
