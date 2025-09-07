from __future__ import annotations

from typing import Tuple

import torch


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    # x: [B, H, W, C] -> [B*nW, Ws*Ws, C]
    B, H, W, C = x.shape
    Ws = int(window_size)
    x = x.view(B, H // Ws, Ws, W // Ws, Ws, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, Ws * Ws, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int, B: int) -> torch.Tensor:
    # windows: [B*nW, Ws*Ws, C] -> [B, H, W, C]
    Ws = int(window_size)
    C = windows.shape[-1]
    x = windows.view(B, H // Ws, W // Ws, Ws, Ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    return x


def roll_spatial(x: torch.Tensor, shifts: Tuple[int, int]) -> torch.Tensor:
    # x: [B,H,W,C]
    sH, sW = int(shifts[0]), int(shifts[1])
    if sH == 0 and sW == 0:
        return x
    return torch.roll(x, shifts=(sH, sW), dims=(1, 2))
