from __future__ import annotations

"""
DINOv3 + ConvGRU + FPN pipeline

- Keeps the existing encoder->ConvGRU flow, but *inserts ConvGRU between the
  1x1 projection and the FPN*, per the user's spec.
- Uses torchvision.ops.FeaturePyramidNetwork (no custom FPN reimplementation).
- ConvGRU is initialized to behave like (approximate) identity at init time.
- Provides a sequence wrapper that accepts (B, T, 3, H, W) and emits heatmaps at
  output_stride, fusing FPN levels to a single-map head.

Main classes:
- ConvGRUCell: ConvGRU with identity-like initialization
- DINOv3ViTGRUAdapter: ViT -> tokens -> 1x1 proj -> ConvGRU -> FPN (dict{"0","1","2"})
- FPNMergeHead: merge FPN maps (upsample to highest-res, concat, predict 1ch)
- SequenceHeatmapNet: loops over time; calls adapter per step; applies head; upsamples to output_stride
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork


# -----------------------------------------------------------------------------
# ConvGRU with identity-like initialization
# -----------------------------------------------------------------------------
class ConvGRUCell(nn.Module):
    """ConvGRU cell whose initial behavior approximates identity mapping.

    Formulation:
      gates = Conv([x, h]) -> split into reset r, update z
      h_tilde = tanh(Conv([x, r*h]))
      h_next = (1 - z) * h + z * h_tilde

    Identity-ish init is achieved by setting:
      - conv_gates.weight = 0, conv_gates.bias_update << 0 (e.g., -5)
      - conv_can.weight = 0, conv_can.bias = 0
    so that z ~ 0 initially and h_next ~ h.
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        pad = kernel_size // 2
        self.conv_gates = nn.Conv2d(input_dim + hidden_dim, hidden_dim * 2, kernel_size, padding=pad)
        self.conv_can = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=pad)
        self._init_identity()

    @torch.no_grad()
    def _init_identity(self) -> None:
        # gates: [reset, update]
        nn.init.zeros_(self.conv_gates.weight)
        nn.init.zeros_(self.conv_gates.bias)
        # make update gate ~0 via strong negative bias
        self.conv_gates.bias[self.hidden_dim :] = -5.0  # sigmoid(-5) ~= 0.0067
        # candidate state has no effect initially
        nn.init.zeros_(self.conv_can.weight)
        nn.init.zeros_(self.conv_can.bias)

    def forward(self, x: torch.Tensor, h_cur: torch.Tensor) -> torch.Tensor:
        g = self.conv_gates(torch.cat([x, h_cur], dim=1))
        r, z = g.chunk(2, dim=1)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        h_tilde = torch.tanh(self.conv_can(torch.cat([x, r * h_cur], dim=1)))
        h_next = (1.0 - z) * h_cur + z * h_tilde
        return h_next


# -----------------------------------------------------------------------------
# DINOv3 ViT Adapter with ConvGRU inserted before FPN
# -----------------------------------------------------------------------------
class DINOv3ViTGRUAdapter(nn.Module):
    """Adapt a (optionally frozen) DINOv3 ViT into multi-scale maps via ConvGRU + FPN.

    Flow:
      ViT -> patch tokens -> BCHW map -> 1x1 projection (C_embed -> C_out)
         -> ConvGRU(h) on the projected map (time-aware)
         -> build coarse maps (x2, x4 downsample) -> FPN -> {"0","1","2"}

    Notes:
      - If `freeze=True`, gradients through ViT are disabled and params have requires_grad=False.
      - ConvGRU keeps channel count: input_dim = hidden_dim = out_channels.
      - Forward can be called per-time-step with an optional hidden state `h_cur` and
        returns (fpn_feats, h_next).
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

        # Infer patch size & feature dim
        patch_size = getattr(self.vit, "patch_size", None)
        embed_dim = getattr(self.vit, "embed_dim", None) or getattr(self.vit, "num_features", None)
        if patch_size is None or embed_dim is None:
            raise RuntimeError("Could not infer patch_size/embed_dim from DINOv3 ViT")
        self.patch_size: int = int(patch_size)
        self.embed_dim: int = int(embed_dim)

        self._freeze = bool(freeze)
        if self._freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

        # 1x1 projection from token dim to detector channel dim
        self.proj = nn.Conv2d(self.embed_dim, out_channels, kernel_size=1)

        # ConvGRU (channels preserved)
        self.gru = ConvGRUCell(input_dim=out_channels, hidden_dim=out_channels, kernel_size=3)

        # Coarse feature builders (keep channels; spatial downsample via pooling)
        self.make_coarse4 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.make_coarse5 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # TorchVision FPN (no custom reimplementation)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[out_channels, out_channels, out_channels],
            out_channels=out_channels,
        )

        self.out_channels = out_channels
        self._fpn_levels = max(1, int(fpn_levels))

    def _vit_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # Respect freeze flag for autograd context
        with torch.no_grad() if self._freeze else torch.enable_grad():
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

    def forward(
        self, x: torch.Tensor, h_cur: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Run one step. Returns (fpn_feats, h_next).

        Args:
          x: [B, 3, H, W] input frame
          h_cur: optional hidden state [B, C_out, H/patch, W/patch]. If None, zeros are used.
        """
        tokens = self._vit_patch_tokens(x)
        b, n, c = tokens.shape
        h = x.shape[-2] // self.patch_size
        w = x.shape[-1] // self.patch_size
        if n != h * w:
            raise AssertionError(f"Token count {n} mismatch with HxW {h}x{w} (patch={self.patch_size})")
        feat_2d = tokens.transpose(1, 2).contiguous().view(b, c, h, w)  # [B, C, H/ps, W/ps]

        c3 = self.proj(feat_2d)  # [B, out_ch, H/ps, W/ps]

        if h_cur is None or h_cur.shape != c3.shape:
            # initialize hidden with zeros (on the same device/dtype)
            h_cur = torch.zeros_like(c3)

        h_next = self.gru(c3, h_cur)  # ConvGRU inserted here

        # Downsample to coarse maps, then lateral 1x1s
        c4 = self.pool2(h_next)
        c5 = self.pool2(c4)

        lat3 = h_next
        lat4 = self.make_coarse4(c4)
        lat5 = self.make_coarse5(c5)

        # keys "0","1","2" as torchvision FPN convention
        fpn_out = self.fpn({"0": lat3, "1": lat4, "2": lat5})
        return fpn_out, h_next


# -----------------------------------------------------------------------------
# Merge head for FPN outputs -> 1ch heatmap (at adapter's P3 stride)
# -----------------------------------------------------------------------------
class FPNMergeHead(nn.Module):
    """Merge FPN maps by upsampling to the highest-res ("0"), then predict 1ch map.

    This head does not implement an FPN; it merely fuses the FPN outputs.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.smooth0 = nn.Conv2d(channels, channels, 3, padding=1)
        self.smooth1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.smooth2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.merge = nn.Conv2d(channels * 3, channels, 1)
        self.pred = nn.Conv2d(channels, 1, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        p0 = self.smooth0(feats["0"])  # highest spatial resolution
        p1 = self.smooth1(F.interpolate(feats["1"], size=p0.shape[-2:], mode="bilinear", align_corners=False))
        p2 = self.smooth2(F.interpolate(feats["2"], size=p0.shape[-2:], mode="bilinear", align_corners=False))
        x = torch.cat([p0, p1, p2], dim=1)
        x = self.act(self.merge(x))
        x = self.pred(x)
        return x


# -----------------------------------------------------------------------------
# Sequence wrapper: (B, T, 3, H, W) -> (B, T, 1, H/out_stride, W/out_stride)
# -----------------------------------------------------------------------------
@dataclass
class NetConfig:
    repo_dir: str
    entry: str
    weights: str
    out_channels: int = 256
    fpn_levels: int = 3
    freeze_backbone: bool = True
    output_stride: int = 4  # heatmap stride w.r.t. input


class SequenceHeatmapNet(nn.Module):
    def __init__(self, cfg: NetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.adapter = DINOv3ViTGRUAdapter(
            repo_dir=cfg.repo_dir,
            entry=cfg.entry,
            weights=cfg.weights,
            out_channels=cfg.out_channels,
            fpn_levels=cfg.fpn_levels,
            freeze=cfg.freeze_backbone,
        )
        self.head = FPNMergeHead(cfg.out_channels)

    @property
    def patch_size(self) -> int:
        return self.adapter.patch_size

    def _upsample_to_output_stride(self, x: torch.Tensor) -> torch.Tensor:
        # Adapter highest-res ("0") is at stride = patch_size.
        # Desired final stride = output_stride, so scale factor = patch_size / output_stride.
        scale = self.patch_size / float(self.cfg.output_stride)
        if abs(scale - round(scale)) < 1e-6:  # integer scale preferred
            scale = int(round(scale))
            return F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
        # Fallback: compute target spatial size from input * exact ratio
        h, w = x.shape[-2:]
        th, tw = int(round(h * scale)), int(round(w * scale))
        return F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Args:
          x: (B, T, 3, H, W) or (B, 3, H, W)
          h0: optional initial hidden state for ConvGRU at P3 resolution
        Returns:
          heatmaps: (B, T, 1, H/out_stride, W/out_stride)
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B, 1, C, H, W)
        b, t, c, h, w = x.shape

        # initialize hidden if needed using the first frame shape
        h_cur: Optional[torch.Tensor] = None if h0 is None else h0

        outs = []
        for i in range(t):
            feats, h_cur = self.adapter(x[:, i], h_cur)
            heat = self.head(feats)  # (B,1,H/ps,W/ps)
            heat = self._upsample_to_output_stride(heat)
            outs.append(heat)
        return torch.stack(outs, dim=1)


# Convenience builder ---------------------------------------------------------


def build_model(cfg_dict: dict) -> SequenceHeatmapNet:
    cfg = NetConfig(**cfg_dict)
    return SequenceHeatmapNet(cfg)
