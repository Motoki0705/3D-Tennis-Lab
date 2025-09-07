from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .rel_pos_bias import RelativePositionBias2D


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rpb = RelativePositionBias2D((window_size, window_size), num_heads)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B*nW, Ws*Ws, C]
        Bnw, N, C = x.shape
        qkv = self.qkv(x).reshape(Bnw, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [Bnw,h,N,N]
        attn = attn + self.rpb()[None, :, :, :]
        if attn_mask is not None:
            # attn_mask: [nW, N, N] -> broadcast to [B,nW,h,N,N]
            nW = attn_mask.shape[0]
            attn = attn.view(-1, nW, self.num_heads, N, N)
            attn = attn + attn_mask[:, None, :, :]
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        y = (attn @ v).transpose(1, 2).reshape(Bnw, N, C)
        y = self.proj_drop(self.proj(y))
        return y
