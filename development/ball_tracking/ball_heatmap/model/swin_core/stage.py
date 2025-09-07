from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .swin_block import SwinTransformerBlock
from .patch_merging import PatchMerging


class SwinStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: List[float] | float = 0.0,
        downsample: bool = False,
        use_moe: bool = False,
        moe_blocks: Optional[List[int]] = None,
        build_moe_ffn: Optional[callable] = None,
    ):
        super().__init__()
        if isinstance(drop_path, float):
            drop_path = [drop_path] * depth
        assert len(drop_path) == depth
        blocks: List[nn.Module] = []
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else window_size // 2
            use_block_moe = use_moe and (moe_blocks is not None and i in moe_blocks)
            moe_ffn = build_moe_ffn() if (use_block_moe and build_moe_ffn is not None) else None
            blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                    use_moe=use_block_moe,
                    moe_ffn=moe_ffn,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.downsample = PatchMerging(dim) if downsample else None
        self.out_dim = dim * 2 if downsample else dim

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        for blk in self.blocks:
            x = blk(x, H, W)
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        return x, H, W
