from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import WindowAttention
from .window_ops import window_partition, window_reverse, roll_spatial
from .drop_path import DropPath


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        use_moe: bool = False,
        moe_ffn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size=self.window_size, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        # by default use given moe_ffn else simple MLP
        if use_moe and moe_ffn is not None:
            self.ffn = moe_ffn
        else:
            from .mlp import Mlp

            self.ffn = Mlp(dim, hidden, drop=drop)

    def _build_attn_mask(self, H: int, W: int, device) -> Optional[torch.Tensor]:
        if self.shift_size == 0:
            return None
        Ws = self.window_size
        img_mask = torch.zeros((1, H, W, 1), device=device)
        cnt = 0
        for h in (slice(0, -Ws), slice(-Ws, -self.shift_size), slice(-self.shift_size, None)):
            for w in (slice(0, -Ws), slice(-Ws, -self.shift_size), slice(-self.shift_size, None)):
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, Ws).view(-1, Ws * Ws)
        attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf")).masked_fill(attn_mask == 0, 0.0)
        return attn_mask  # [nW, N, N]

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        short = x
        x = self.norm1(x).view(B, H, W, C)

        # Pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))  # (B, H, W, C) -> pad C, W, H
        _, H_pad, W_pad, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted = roll_spatial(x, (-self.shift_size, -self.shift_size))
            attn_mask = self._build_attn_mask(H_pad, W_pad, x.device)
        else:
            shifted = x
            attn_mask = None

        # window partition
        windows = window_partition(shifted, self.window_size)  # [B*nW, Ws*Ws, C]
        attn_windows = self.attn(windows, attn_mask)

        # window reverse
        shifted_back = window_reverse(attn_windows, self.window_size, H_pad, W_pad, B)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = roll_spatial(shifted_back, (self.shift_size, self.shift_size))
        else:
            x = shifted_back

        # Unpad feature maps
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = short + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x
