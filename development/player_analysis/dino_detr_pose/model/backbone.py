"""DINOv3 backbone adapter producing multi-scale features for DETRPose."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Dinov3BackboneConfig:
    repo_dir: str = "third_party/dinov3"
    entry: str = "dinov3_vits16"
    weights: Optional[str] = "third_party/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    out_channels: int = 256
    freeze: bool = True


class Dinov3PoseBackbone(nn.Module):
    """Wrap a DINOv3 ViT and emit stride {8,16,32} feature maps."""

    def __init__(self, cfg: Dinov3BackboneConfig) -> None:
        super().__init__()
        repo_dir = Path(cfg.repo_dir)
        if cfg.weights is not None:
            weights_path = Path(cfg.weights)
            weights = str(weights_path) if weights_path.exists() else cfg.weights
        else:
            weights = None
        self.vit = torch.hub.load(str(repo_dir), cfg.entry, source="local", weights=weights)
        self.freeze = bool(cfg.freeze)

        self.patch_size = int(getattr(self.vit, "patch_size", 16))
        embed_dim = int(getattr(self.vit, "embed_dim", getattr(self.vit, "num_features", 768)))
        self.proj = nn.Conv2d(embed_dim, cfg.out_channels, kernel_size=1)
        self.up_proj = nn.Conv2d(cfg.out_channels, cfg.out_channels, kernel_size=3, padding=1)
        self.down_proj = nn.Conv2d(cfg.out_channels, cfg.out_channels, kernel_size=3, stride=2, padding=1)
        self.smooth = nn.Conv2d(cfg.out_channels, cfg.out_channels, kernel_size=3, padding=1)

        if self.freeze:
            for param in self.vit.parameters():
                param.requires_grad_(False)

        self.out_channels = cfg.out_channels
        self.feat_strides = (8, 16, 32)

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:  # [B,3,H,W]
        tokens = self._extract_patch_tokens(images)
        b, n, c = tokens.shape
        h = int(images.shape[-2] // self.patch_size)
        w = int(images.shape[-1] // self.patch_size)
        feat_2d = tokens.transpose(1, 2).reshape(b, c, h, w)

        base = self.proj(feat_2d)
        base = self.smooth(F.gelu(base))

        # stride 16 output directly
        p16 = base
        # stride 8 via upsample
        p8 = self.up_proj(F.interpolate(base, scale_factor=2.0, mode="bilinear", align_corners=False))
        # stride 32 via downsample
        p32 = self.down_proj(base)

        return [p8, p16, p32]

    def _extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad() if self.freeze else torch.enable_grad():
            feats = self.vit.forward_features(x)
        if isinstance(feats, dict) and "x_norm_patchtokens" in feats:
            return feats["x_norm_patchtokens"]
        if isinstance(feats, dict) and "x_prenorm" in feats:
            return feats["x_prenorm"]
        if isinstance(feats, (list, tuple)) and feats:
            cand = feats[-1]
            if isinstance(cand, dict) and "x_norm_patchtokens" in cand:
                return cand["x_norm_patchtokens"]
        raise RuntimeError("Unexpected DINOv3 forward_features output structure.")


__all__ = ["Dinov3PoseBackbone", "Dinov3BackboneConfig"]
