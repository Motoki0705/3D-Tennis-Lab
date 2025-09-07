from __future__ import annotations

import torch
import torch.nn as nn


class PatchMerging(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int) -> tuple[torch.Tensor, int, int]:
        # x: [B, H*W, C] -> [B, H/2*W/2, 2C]
        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :], x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]], dim=-1)
        x = x.view(B, -1, 4 * C)
        x = self.reduction(x)
        return x, H // 2, W // 2
