from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from development.ball_tracking.ball_heatmap.model.temporal.multiscale import MultiScaleTemporal


class PixelShuffleUp(nn.Module):
    """
    Learnable upsampling using 1x1 conv + pixel shuffle.
    Assumes integer scale factor between feature maps (typically 2x in FPN).
    Falls back to nearest if factor is 1.
    """

    def __init__(self, channels: int, factor: int = 2):
        super().__init__()
        self.factor = int(factor)
        if self.factor > 1:
            self.proj = nn.Conv2d(channels, channels * (self.factor**2), kernel_size=1, bias=False)
            self.ps = nn.PixelShuffle(self.factor)
            self.bn = nn.BatchNorm2d(channels)
            self.act = nn.ReLU(inplace=True)
        else:
            # identity path
            self.proj = None

    def forward(self, x: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        if self.factor <= 1:
            return x
        # Validate factor
        h, w = x.shape[-2:]
        th, tw = target_hw
        # If target is not exact multiple, approximate by rounding
        fh = max(1, th // h)
        fw = max(1, tw // w)
        f = min(fh, fw)
        if f != self.factor:
            # Reconfigure on the fly if factor mismatch (rare)
            # Note: this keeps weights separate per call; acceptable for static FPN patterns.
            proj = nn.Conv2d(x.shape[1], x.shape[1] * (f**2), kernel_size=1, bias=False).to(x.device, x.dtype)
            y = nn.PixelShuffle(f)(proj(x))
            return y
        y = self.ps(self.proj(x))
        y = self.act(self.bn(y))
        return y


class FPNDecoder(nn.Module):
    """
    FPN-like decoder with learnable pixel-shuffle upsampling in top-down pathway.
    - Input: dict {stride: [B*C, C_s, H_s, W_s]}
    - Lateral: 1x1 conv per input
    - Top-down: pixel-shuffle upsample and sum
    - Output: dict {stride: [B*C, C_out, H_s, W_s]} for requested strides
    """

    def __init__(
        self,
        in_channels_map: Dict[int, int],
        out_channels: int,
        use_strides: List[int],
        temporal_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.use_strides = sorted(list(use_strides), reverse=True)  # high->low (e.g., 32,16,8,4)
        self.out_channels = int(out_channels)

        # Lateral 1x1 convs per available input stride
        self.laterals = nn.ModuleDict({
            str(s): nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False) for s, in_ch in in_channels_map.items()
        })

        # Top-down pixel-shuffle upsampler (assume 2x between pyramid levels)
        self.upsample = PixelShuffleUp(out_channels, factor=2)

        # Post-merge smoothing conv per output stride
        self.smooth = nn.ModuleDict({
            str(s): nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            for s in self.use_strides
        })

        # Optional temporal block over decoder outputs. Because all outputs share the same
        # channel size (out_channels), we can safely share the temporal block across scales.
        self.temporal_enabled = False
        self.temporal_block: Optional[MultiScaleTemporal] = None
        if temporal_cfg is not None and bool(temporal_cfg.get("enabled", False)):
            cfg = {k: v for k, v in temporal_cfg.items()}
            self.temporal_block = MultiScaleTemporal(
                strides=[int(s) for s in self.use_strides],
                channels=self.out_channels,
                cfg=cfg,
                share_across_scales=True,
            )
            self.temporal_enabled = True

    def forward(self, feats: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        # Build pyramid dict of laterals
        lat = {s: self.laterals[str(s)](feats[s]) for s in feats.keys() if str(s) in self.laterals}

        # Process in top-down order, summing element-wise
        out: Dict[int, torch.Tensor] = {}
        prev: torch.Tensor | None = None
        for s in sorted(lat.keys(), reverse=True):  # high->low
            x = lat[s]
            if prev is not None:
                # pixel-shuffle upsample previous to current resolution and add
                prev_up = self.upsample(prev, target_hw=x.shape[-2:])
                # In case off-by-one spatial due to factor mismatch, pad/crop to match
                if prev_up.shape[-2:] != x.shape[-2:]:
                    th, tw = x.shape[-2:]
                    ph, pw = prev_up.shape[-2:]
                    prev_up = prev_up[:, :, : min(ph, th), : min(pw, tw)]
                    x = x[:, :, : min(ph, th), : min(pw, tw)]
                x = x + prev_up
            out[s] = self.smooth[str(s)](x)
            prev = out[s]

        # Filter to requested strides, preserving spatial sizes
        return {s: out[s] for s in self.use_strides if s in out}

    def forward_bt(self, feats: Dict[int, torch.Tensor], B: int, T: int) -> Dict[int, torch.Tensor]:
        """
        Sequence-aware forward when encoder features are flattened as [B*T,C,H,W].
        Applies FPN merge + smooth, then temporal per-scale if configured, and flattens back to [B*T,C,H,W].
        """
        out = self.forward(feats)
        if not (self.temporal_enabled and self.temporal_block is not None and T > 1):
            return out
        # reshape per-scale to [B,T,C,H,W]
        bt = {s: f.view(B, T, *f.shape[1:]) for s, f in out.items()}
        bt = self.temporal_block(bt)
        return {s: f.reshape(B * T, *f.shape[2:]) for s, f in bt.items()}
