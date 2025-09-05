from __future__ import annotations

from typing import Dict, List, Union

import torch
import torch.nn as nn

from .factory import build_temporal_block


class MultiScaleTemporal(nn.Module):
    """
    Apply the same or per-scale temporal block to {stride: [B,T,C,H,W]}.
    All inputs must share the same channel dimension C.
    """

    def __init__(
        self,
        strides: List[int],
        channels: Union[int, Dict[int, int]],
        cfg: dict,
        share_across_scales: bool = True,
    ):
        super().__init__()
        self.strides = list(strides)
        self.share = bool(share_across_scales)
        if self.share:
            assert isinstance(channels, int), "Shared multi-scale temporal requires a single channel size."
            self.block = build_temporal_block(cfg, int(channels))
        else:
            # Independent blocks per scale; allow per-scale channels
            if isinstance(channels, dict):
                ch_map = {int(k): int(v) for k, v in channels.items()}
            else:
                ch_map = {int(s): int(channels) for s in self.strides}
            self.blocks = nn.ModuleDict({str(s): build_temporal_block(cfg, ch_map[int(s)]) for s in self.strides})

    def forward(self, feats: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        out: Dict[int, torch.Tensor] = {}
        if self.share:
            for s in self.strides:
                out[s] = self.block(feats[s])
        else:
            for s in self.strides:
                out[s] = self.blocks[str(s)](feats[s])
        return out
