from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torchvision.ops import FeaturePyramidNetwork


class DINOv3ViTAdapter(nn.Module):
    """Adapt a frozen DINOv3 ViT into multi-scale feature maps for heatmap detection.

    Mirrors the structure used in player_analysis.dino_faster_rcnn: loads a ViT via
    local torch.hub (`third_party/dinov3`), freezes it, reshapes patch tokens into
    2D maps, projects channels, and builds a small FPN.
    """

    def __init__(
        self,
        *,
        repo_dir: str,
        entry: str,
        weights: str,
        out_channels: int = 256,
        fpn_levels: int = 3,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.vit = torch.hub.load(repo_dir, entry, source="local", weights=weights)

        # Determine patch size and embedding dim from DINOv3 ViT
        patch_size = getattr(self.vit, "patch_size", None)
        embed_dim = getattr(self.vit, "embed_dim", None) or getattr(self.vit, "num_features", None)
        if patch_size is None or embed_dim is None:
            raise RuntimeError("Could not infer patch_size/embed_dim from DINOv3 ViT")
        self.patch_size: int = int(patch_size)
        self.embed_dim: int = int(embed_dim)

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

        # 1x1 projection from token dim to detector channel dim
        self.proj = nn.Conv2d(self.embed_dim, out_channels, kernel_size=1)

        # Build a minimal FPN over 3 levels by downsampling the base map
        self.make_coarse4 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.make_coarse5 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[out_channels, out_channels, out_channels], out_channels=out_channels
        )

        self.out_channels = out_channels
        self._fpn_levels = max(1, int(fpn_levels))

    @torch.no_grad()
    def _vit_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.vit.forward_features(x)
        if not isinstance(feats, dict):
            if isinstance(feats, list) and len(feats) > 0 and isinstance(feats[0], dict):
                feats = feats[0]
            else:
                raise RuntimeError("Unexpected DINOv3 feature output format")
        tokens = feats.get("x_norm_patchtokens", None)
        if tokens is None:
            raise RuntimeError("DINOv3 features missing 'x_norm_patchtokens'")
        return tokens  # [B, N, C]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract ViT patch tokens and reshape into BCHW
        with torch.no_grad():
            tokens = self._vit_patch_tokens(x)
        b, n, c = tokens.shape
        h = x.shape[-2] // self.patch_size
        w = x.shape[-1] // self.patch_size
        assert n == h * w, f"Token count {n} mismatch with HxW {h}x{w} (patch={self.patch_size})"
        feat_2d = tokens.transpose(1, 2).contiguous().view(b, c, h, w)

        c3 = self.proj(feat_2d)
        c4 = nn.functional.max_pool2d(c3, kernel_size=2, stride=2)
        c5 = nn.functional.max_pool2d(c4, kernel_size=2, stride=2)

        lat3 = c3
        lat4 = self.make_coarse4(c4)
        lat5 = self.make_coarse5(c5)

        # Use keys "0","1","2" as in torchvision's FPN convention
        fpn_out = self.fpn({"0": lat3, "1": lat4, "2": lat5})
        return fpn_out


def build_dinov3_vit_backbone(backbone_cfg: dict) -> DINOv3ViTAdapter:
    return DINOv3ViTAdapter(
        repo_dir=backbone_cfg.get("repo_dir", "third_party/dinov3"),
        entry=backbone_cfg.get("entry", "dinov3_vits16"),
        weights=backbone_cfg.get("weights", "third_party/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
        out_channels=int(backbone_cfg.get("out_channels", 256)),
        fpn_levels=int(backbone_cfg.get("fpn_levels", 3)),
        freeze=bool(backbone_cfg.get("freeze", True)),
    )


__all__ = ["DINOv3ViTAdapter", "build_dinov3_vit_backbone"]
