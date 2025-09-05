from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalMSA(nn.Module):
    """
    Temporal self-attention across time only (divided space-time).
    Applies attention per spatial location across frames.

    Input:  x [B, T, C, H, W]
    Output: y [B, T, C, H, W]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        window_T: int = 5,
        causal: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.h = num_heads
        self.d = dim // num_heads
        self.wT = int(window_T) if window_T is not None else 0
        self.causal = bool(causal)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x_lin = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        qkv = self.qkv(x_lin).chunk(3, dim=-1)
        q = qkv[0].view(-1, T, self.h, self.d).transpose(1, 2)
        k = qkv[1].view(-1, T, self.h, self.d).transpose(1, 2)
        v = qkv[2].view(-1, T, self.h, self.d).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.d**0.5)
        if self.causal:
            idx = torch.arange(T, device=x.device)
            mask_future = idx[None, None, :, None] < idx[None, None, None, :]
            att = att.masked_fill(mask_future, float("-inf"))
        if self.wT and self.wT < T:
            idx = torch.arange(T, device=x.device)
            dist = (idx[None, :, None] - idx[None, None, :]).abs()
            band = dist > (self.wT // 2)
            att = att.masked_fill(band, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(-1, T, C)
        y = self.proj_drop(self.proj(y))
        y = y.view(B, H, W, T, C).permute(0, 3, 4, 1, 2).contiguous()
        return y
