from __future__ import annotations

import torch
import torch.nn as nn


class HeatmapHead(nn.Module):
    """Simple upsampling head to produce 1ch heatmap at stride=output_stride.

    - Takes highest-res FPN map (P3 ~ stride 16 for ViT-S/16 input) and upsamples by 4x
      to reach stride 4 by default.
    - Two Conv blocks for refinement, then final 1x1 conv for logits.
    """

    def __init__(self, in_channels: int, mid_channels: int = 256, upsample_factor: int = 4):
        super().__init__()
        self.upsample_factor = int(upsample_factor)

        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(mid_channels, 1, kernel_size=1)

    def forward(self, fpn_feats: dict[str, torch.Tensor]) -> torch.Tensor:
        # Use highest-resolution FPN level (key "0" from our adapter)
        p3 = fpn_feats["0"]
        x = self.refine(p3)
        if self.upsample_factor != 1:
            x = nn.functional.interpolate(x, scale_factor=self.upsample_factor, mode="bilinear", align_corners=False)
        logits = self.head(x)
        return logits  # (B,1,H/stride,W/stride)


__all__ = ["HeatmapHead"]
