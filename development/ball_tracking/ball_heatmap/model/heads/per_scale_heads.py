from typing import Dict, List

import torch
import torch.nn as nn

# -------------------- Heads --------------------


class HeatmapHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SpeedHeadGlobal(nn.Module):
    """Predict per-frame screen-normalized velocity vector (dx/W, dy/H) as [B*T, 2]."""

    def __init__(self, in_channels: int, hidden: int = 256, out_channels: int = 2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B*T, C, H, W] (flatten time beforehand)
        x = self.pool(feat)  # [B*T, C, 1, 1]
        x = self.fc(x).squeeze(-1).squeeze(-1)  # [B*T, 2]
        return x


class VisHeadGlobal(nn.Module):
    """Predict per-frame COCO visibility logits (3 classes) as [B*T, 3]."""

    def __init__(self, in_channels: int, hidden: int = 256, num_classes: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, num_classes, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.pool(feat)  # [B*T, C, 1, 1]
        x = self.fc(x).squeeze(-1).squeeze(-1)  # [B*T, 3]
        return x


class PerScaleHeads(nn.Module):
    """
    Produce heatmaps per scale, and global (per-frame) speed & vis logits from the finest-scale feature.
    Input: dict {stride: Tensor[B*C?, C, H, W]}  (time may be flattened as B*T)
    Output:
      - heatmaps: List[Tensor[B*C?, 1, Hs, Ws]] aligned to self.strides (ascending)
      - speed:    Tensor[B*C?, 2]
      - vis:      Tensor[B*C?, 3]
    """

    def __init__(
        self,
        in_channels: int,
        strides: List[int],
        heatmap_channels: int = 1,
        speed_out_channels: int = 2,
        vis_classes: int = 3,
        head_hidden: int = 256,
    ):
        super().__init__()
        self.strides = sorted(list(strides))  # ensure ascending: smallest stride first
        self.hmap = nn.ModuleDict({str(s): HeatmapHead(in_channels, heatmap_channels) for s in self.strides})
        # global heads use the finest-scale (smallest stride) features
        self.speed_head = SpeedHeadGlobal(in_channels, hidden=head_hidden, out_channels=speed_out_channels)
        self.vis_head = VisHeadGlobal(in_channels, hidden=head_hidden, num_classes=vis_classes)

    def forward(self, feats_by_s: Dict[int, torch.Tensor]):
        # heatmaps per scale
        heatmaps = [self.hmap[str(s)](feats_by_s[s]) for s in self.strides]

        # pick finest scale feature (smallest stride)
        s0 = self.strides[0]
        finest = feats_by_s[s0]  # expected [B*T, C, H, W] or [B, C, H, W] (time pre-flattened upstream)

        # if [B, C, H, W], treat as [B*1, C, H, W] – heads are per frame either way
        if finest.dim() == 5:
            # If a decoder returns [B, T, C, H, W] (rare), flatten here
            B, T = finest.shape[:2]
            finest = finest.view(B * T, *finest.shape[2:])

        speed = self.speed_head(finest)  # [B*T, 2] or [B,2]
        vis_logits = self.vis_head(finest)  # [B*T, 3] or [B,3]
        return heatmaps, speed, vis_logits
