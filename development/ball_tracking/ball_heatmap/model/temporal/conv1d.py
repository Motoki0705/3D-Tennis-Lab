from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TemporalBlock


class TemporalConv1D(TemporalBlock):
    """
    Depthwise temporal Conv3d over time dimension (cheap, parallel).
    """

    def __init__(self, channels: int, kernel_t: int = 3, causal: bool = False, expand: int = 1):
        super().__init__()
        assert kernel_t % 2 == 1 or causal, "odd kernel unless causal"
        self.causal = bool(causal)
        self.kernel_t = int(kernel_t)

        # depthwise temporal conv (groups=C) on [B,C,T,H,W] => Conv3d with (k,1,1)
        padding = (kernel_t // 2, 0, 0)
        self.dw = nn.Conv3d(channels, channels, (kernel_t, 1, 1), padding=padding, groups=channels, bias=False)
        mid = channels * int(expand)
        self.pw1 = nn.Conv3d(channels, mid, 1, bias=False)
        self.bn = nn.BatchNorm3d(mid)
        self.act = nn.ReLU(inplace=True)
        self.pw2 = nn.Conv3d(mid, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C,H,W] -> [B,C,T,H,W]
        x5 = x.permute(0, 2, 1, 3, 4)
        if self.causal:
            # left pad only in time dimension
            pad_t = (self.kernel_t - 1, 0)
            x5 = F.pad(x5, (0, 0, 0, 0, pad_t[0], pad_t[1]))
        y = self.dw(x5)
        y = self.pw2(self.act(self.bn(self.pw1(y))))
        return y.permute(0, 2, 1, 3, 4)
