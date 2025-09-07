from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class RelativePositionBias2D(nn.Module):
    def __init__(self, window_size: Tuple[int, int], num_heads: int):
        super().__init__()
        Wh, Ww = int(window_size[0]), int(window_size[1])
        self.window_size = (Wh, Ww)
        self.num_heads = int(num_heads)
        self.num_relative = (2 * Wh - 1) * (2 * Ww - 1)
        self.table = nn.Parameter(torch.zeros(self.num_relative, num_heads))
        self.register_buffer("index", self._build_index(Wh, Ww), persistent=False)
        nn.init.trunc_normal_(self.table, std=0.02)

    @staticmethod
    def _build_index(Wh: int, Ww: int) -> torch.Tensor:
        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # [2,Wh,Ww]
        coords_flatten = torch.flatten(coords, 1)  # [2, Wh*Ww]
        rel_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Wh*Ww, Wh*Ww]
        rel_coords = rel_coords.permute(1, 2, 0).contiguous()  # [Wh*Ww, Wh*Ww, 2]
        rel_coords[:, :, 0] += Wh - 1
        rel_coords[:, :, 1] += Ww - 1
        rel_coords[:, :, 0] *= 2 * Ww - 1
        rel_index = rel_coords.sum(-1)  # [Wh*Ww, Wh*Ww]
        return rel_index

    def forward(self) -> torch.Tensor:
        # [num_heads, Wh*Ww, Wh*Ww]
        bias = self.table[self.index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], self.num_heads
        )
        return bias.permute(2, 0, 1).contiguous()
