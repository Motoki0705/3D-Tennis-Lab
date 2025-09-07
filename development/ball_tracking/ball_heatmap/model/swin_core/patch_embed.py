from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_channels: int = 3, embed_dim: int = 96, patch_size: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(img_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.H0 = 0
        self.W0 = 0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        # x: [B,3,H,W] -> [B, C, H0, W0]
        x = self.proj(x)
        B, C, H0, W0 = x.shape
        self.H0, self.W0 = H0, W0
        tokens = x.flatten(2).transpose(1, 2)  # [B, H0*W0, C]
        tokens = self.norm(tokens)
        return tokens, H0, W0
