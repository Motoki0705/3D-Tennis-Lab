# preset_a_liteunet_context.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Small attention utilities
# ---------------------------


class TinySE(nn.Module):
    """Tiny Squeeze-Excite (channel) with reduction."""

    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        mid = max(4, ch // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, mid, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, ch, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.avg(x))
        return x * w


class ECA(nn.Module):
    """ECA: Efficient Channel Attention (channel-wise 1D conv on pooled vec)."""

    def __init__(self, ch: int, k: int = 5):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k // 2), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg(x)  # [B,C,1,1]
        y = self.conv1d(y.squeeze(-1).transpose(1, 2))  # [B,1,C] -> conv1d -> [B,1,C]
        y = torch.sigmoid(y.transpose(1, 2).unsqueeze(-1))  # [B,C,1,1]
        return x * y


# ---------------------------
# Lightweight core blocks
# ---------------------------


class IRDW(nn.Module):
    """
    Inverted Residual + Depthwise (MobileNetV2-ish).
    1x1 expand → DW3x3 → 1x1 project (+ optional ECA/SE).
    """

    def __init__(self, in_ch: int, out_ch: int, exp: int = 3, se: bool = False, eca: bool = False):
        super().__init__()
        mid = max(out_ch, int(in_ch * exp))
        self.expand = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.dw = nn.Conv2d(mid, mid, 3, stride=1, padding=1, groups=mid, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.proj = nn.Conv2d(mid, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.attn = nn.Identity()
        if se:
            self.attn = TinySE(out_ch, reduction=16)
        elif eca:
            self.attn = ECA(out_ch, k=5)
        self.use_res = in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.expand(x)))
        y = self.act(self.bn2(self.dw(y)))
        y = self.bn3(self.proj(y))
        y = self.attn(y)
        return x + y if self.use_res else y


class DownDW(nn.Module):
    """Depthwise stride-2 + pointwise (cheap downsample)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        return self.act(self.bn(x))


class LKDW(nn.Module):
    """Large-kernel depthwise conv + pointwise (residual)."""

    def __init__(self, ch: int, k: int = 21):
        super().__init__()
        p = k // 2
        self.dw = nn.Conv2d(ch, ch, k, padding=p, groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.pw(y)
        y = self.act(self.bn(y))
        return x + y


class GCLite(nn.Module):
    """
    Global Context (light): learns spatial attention map on reduced channels,
    forms weighted global descriptor, re-injects into features.
    """

    def __init__(self, ch: int, r: int = 16):
        super().__init__()
        red = max(4, ch // r)
        self.reduce = nn.Conv2d(ch, red, 1, bias=False)
        self.attn_logits = nn.Conv2d(red, 1, 1, bias=True)
        self.reexpand = nn.Conv2d(ch, ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        m = self.reduce(x)  # [B,red,H,W]
        logits = self.attn_logits(m)  # [B,1,H,W]
        w_sp = torch.softmax(logits.view(b, 1, -1), dim=-1)  # [B,1,HW]
        x_flat = x.view(b, c, -1)  # [B,C,HW]
        ctx = torch.bmm(x_flat, w_sp.transpose(1, 2))  # [B,C,1]
        ctx = ctx.view(b, c, 1, 1)
        return x + self.reexpand(ctx)


class LiteASPP(nn.Module):
    """
    Depthwise ASPP: rates {1, r6, r12, r18} + image pooling branch.
    Returns fused channels == in_ch (keeps shape).
    """

    def __init__(self, ch: int, out_each: Optional[int] = None, rates: Tuple[int, ...] = (1, 6, 12, 18)):
        super().__init__()
        out_each = out_each or (ch // 2)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=r, dilation=r, groups=ch, bias=False),
                nn.Conv2d(ch, out_each, 1, bias=False),
                nn.BatchNorm2d(out_each),
                nn.SiLU(inplace=True),
            )
            for r in rates
        ])
        self.imgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, out_each, 1, bias=False),
            nn.BatchNorm2d(out_each),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Conv2d(out_each * (len(rates) + 1), ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ys = [b(x) for b in self.branches]
        g = self.imgpool(x)
        g = F.interpolate(g, size=x.shape[-2:], mode="nearest")
        y = torch.cat(ys + [g], dim=1)
        y = self.fuse(y)
        return self.act(self.bn(y))


class LightUp(nn.Module):
    """Nearest ×2 → DWConv → PW, skip 1×1, then fuse 1×1."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.skip_proj = nn.Conv2d(skip_ch, out_ch, 1, bias=False)
        self.fuse = nn.Conv2d(out_ch * 2, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.pw(self.dw(x))
        s = self.skip_proj(skip)
        y = torch.cat([x, s], dim=1)
        return self.act(self.bn(self.fuse(y)))


# ---------------------------
# Model definition
# ---------------------------


@dataclass
class PresetAConfig:
    in_channels: int = 3
    num_keypoints: int = 15
    channels: Tuple[int, int, int] = (32, 64, 128)  # C1, C2, C3
    block_expansion: int = 3
    use_se: bool = True
    use_eca: bool = False
    out_stride: int = 4  # supported: 4 or 2
    use_offset_head: bool = False
    offset_per_kpt: int = 2  # (dx, dy)
    deep_supervision: bool = False  # if True, also predict at OS=8 (pre-up) as aux


class LiteUNetContext(nn.Module):
    """
    Preset-A: IRDW encoder → [LKDW -> GCLite -> LiteASPP] context → LightUp decoder.
    Default: OS=4 output heatmaps (and optional offsets).
    """

    def __init__(self, cfg: PresetAConfig):
        super().__init__()
        self.cfg = cfg
        C1, C2, C3 = cfg.channels

        # --- Stem (DW s=2 -> PW) -> Enc1
        self.stem_dw = nn.Conv2d(
            cfg.in_channels, cfg.in_channels, 3, stride=2, padding=1, groups=cfg.in_channels, bias=False
        )
        self.stem_pw = nn.Conv2d(cfg.in_channels, C1, 1, bias=False)
        self.stem_bn = nn.BatchNorm2d(C1)
        self.stem_act = nn.SiLU(inplace=True)

        self.enc1 = nn.Sequential(
            IRDW(C1, C1, exp=cfg.block_expansion, se=cfg.use_se, eca=cfg.use_eca),
            IRDW(C1, C1, exp=cfg.block_expansion, se=cfg.use_se, eca=cfg.use_eca),
        )

        # --- Down2 -> Enc2
        self.down2 = DownDW(C1, C2)
        self.enc2 = nn.Sequential(
            IRDW(C2, C2, exp=cfg.block_expansion, se=cfg.use_se, eca=cfg.use_eca),
            IRDW(C2, C2, exp=cfg.block_expansion, se=cfg.use_se, eca=cfg.use_eca),
        )

        # --- Down3 -> Enc3
        self.down3 = DownDW(C2, C3)
        self.enc3 = nn.Sequential(
            IRDW(C3, C3, exp=cfg.block_expansion, se=cfg.use_se, eca=cfg.use_eca),
            IRDW(C3, C3, exp=cfg.block_expansion, se=cfg.use_se, eca=cfg.use_eca),
            IRDW(C3, C3, exp=cfg.block_expansion, se=cfg.use_se, eca=cfg.use_eca),
        )

        # --- Context at OS=8
        self.lk = LKDW(C3, k=21)
        self.gc = GCLite(C3, r=16)
        self.aspp = LiteASPP(C3, out_each=C3 // 2, rates=(1, 6, 12, 18))

        # --- Decoder
        # Base is OS=8 at enc3/context. To get OS=4: 1 up with skip enc2.
        # To get OS=2: 2 ups (enc2, then enc1).
        self.up3 = LightUp(in_ch=C3, skip_ch=C2, out_ch=C2)  # OS: 8 -> 4
        self.up2 = LightUp(in_ch=C2, skip_ch=C1, out_ch=C1)  # OS: 4 -> 2

        # --- Heads
        K = cfg.num_keypoints
        self.head_os4 = nn.Conv2d(C2, K, 1)  # main head at OS=4
        if cfg.use_offset_head:
            self.off_os4 = nn.Conv2d(C2, K * cfg.offset_per_kpt, 1)

        if cfg.out_stride == 2:
            self.head_os2 = nn.Conv2d(C1, K, 1)
            if cfg.use_offset_head:
                self.off_os2 = nn.Conv2d(C1, K * cfg.offset_per_kpt, 1)

        if cfg.deep_supervision:
            # Auxiliary prediction at OS=8 (pre-up)
            self.head_os8 = nn.Conv2d(C3, K, 1)

        # Sanity on supported out_stride
        assert cfg.out_stride in (4, 2), "Preset-A supports out_stride 4 or 2."

    def _context(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lk(x)
        x = self.gc(x)
        x = self.aspp(x)
        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        """
        Returns:
          {
            'heatmap': (B,K,H/4,W/4) or (B,K,H/2,W/2),
            'offset' (optional): same scale as heatmap with 2*K channels,
            'aux' (optional, if deep_supervision=True): {'os8': (B,K,H/8,W/8)}
          }
        """
        # Encoder
        x = self.stem_act(self.stem_bn(self.stem_pw(self.stem_dw(x))))  # OS: 2
        e1 = self.enc1(x)  # OS: 2

        x = self.down2(e1)  # OS: 4
        e2 = self.enc2(x)  # OS: 4

        x = self.down3(e2)  # OS: 8
        e3 = self.enc3(x)  # OS: 8

        # Context
        c = self._context(e3)  # OS: 8

        out: Dict[str, torch.Tensor | Dict[str, torch.Tensor]] = {}
        aux: Dict[str, torch.Tensor] = {}

        if self.cfg.deep_supervision:
            aux["os8"] = self.head_os8(c)

        # Decoder to OS=4 (main path)
        d4 = self.up3(c, e2)  # OS: 4
        heat_os4 = self.head_os4(d4)
        out["heatmap"] = torch.sigmoid(heat_os4)
        if self.cfg.use_offset_head:
            out["offset"] = self.off_os4(d4)

        # Optionally go to OS=2
        if self.cfg.out_stride == 2:
            d2 = self.up2(d4, e1)  # OS: 2
            heat_os2 = self.head_os2(d2)
            out["heatmap"] = torch.sigmoid(heat_os2)
            if self.cfg.use_offset_head:
                out["offset"] = self.off_os2(d2)

        if self.cfg.deep_supervision:
            out["aux"] = aux

        return out


# ---------------------------
# Factory & example
# ---------------------------


def build_preset_a(
    num_keypoints: int = 15,
    variant: str = "small",  # "nano" | "small" | "base"
    out_stride: int = 4,
    use_offset_head: bool = False,
    deep_supervision: bool = False,
    se: bool = True,
    eca: bool = False,
) -> LiteUNetContext:
    """
    Ready-to-use builder with channel presets.

    nano:  (24, 48,  96)
    small: (32, 64, 128)
    base:  (48, 80, 160)
    """
    presets = {
        "nano": (24, 48, 96),
        "small": (32, 64, 128),
        "base": (48, 80, 160),
    }
    channels = presets.get(variant.lower())
    if channels is None:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(presets.keys())}.")

    cfg = PresetAConfig(
        in_channels=3,
        num_keypoints=num_keypoints,
        channels=channels,
        block_expansion=3,
        use_se=se,
        use_eca=eca,
        out_stride=out_stride,
        use_offset_head=use_offset_head,
        deep_supervision=deep_supervision,
    )
    return LiteUNetContext(cfg)


if __name__ == "__main__":
    # Quick shape test
    device = "cpu"
    B, C, H, W = 2, 3, 512, 288  # e.g., 16:9-ish
    x = torch.randn(B, C, H, W).to(device)

    # OS=4 heatmaps
    model = build_preset_a(
        num_keypoints=15, variant="small", out_stride=4, use_offset_head=True, deep_supervision=True
    ).to(device)
    with torch.no_grad():
        out = model(x)
    print("OS=4 heatmap:", out["heatmap"].shape)  # -> [2, 15, H/4, W/4]
    if "offset" in out:
        print("OS=4 offset:", out["offset"].shape)  # -> [2, 30, H/4, W/4]
    if "aux" in out:
        print("Aux OS=8:", out["aux"]["os8"].shape)  # -> [2, 15, H/8, W/8]

    # OS=2 heatmaps
    model2 = build_preset_a(
        num_keypoints=15, variant="small", out_stride=2, use_offset_head=False, deep_supervision=False
    ).to(device)
    with torch.no_grad():
        out2 = model2(x)
    print("OS=2 heatmap:", out2["heatmap"].shape)  # -> [2, 15, H/2, W/2]
