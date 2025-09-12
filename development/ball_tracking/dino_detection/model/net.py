from __future__ import annotations

import torch
import torch.nn as nn

from .dinov3_fpn import build_dinov3_vit_backbone
from .heatmap_head import HeatmapHead


class DinoHeatmapNet(nn.Module):
    def __init__(self, model_cfg: dict):
        super().__init__()
        backbone_cfg = model_cfg.get("backbone", {})
        head_cfg = model_cfg.get("head", {})

        self.backbone = build_dinov3_vit_backbone(backbone_cfg)
        in_ch = int(backbone_cfg.get("out_channels", 256))
        up = int(head_cfg.get("upsample_factor", 4))
        mid = int(head_cfg.get("mid_channels", 256))
        self.head = HeatmapHead(in_channels=in_ch, mid_channels=mid, upsample_factor=up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fpn_feats = self.backbone(x)
        logits = self.head(fpn_feats)
        return logits


def build_model(cfg_like: dict) -> DinoHeatmapNet:
    return DinoHeatmapNet(cfg_like)


__all__ = ["DinoHeatmapNet", "build_model"]
