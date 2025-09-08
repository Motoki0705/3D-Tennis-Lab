# irdw_attn_fpn.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Core lightweight blocks
# ---------------------------


class TinySE(nn.Module):
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


class IRDW(nn.Module):
    """
    Inverted residual with depthwise 3x3: 1x1 expand -> DW3 -> 1x1 project (+ optional SE).
    """

    def __init__(self, in_ch: int, out_ch: int, exp: int = 3, se: bool = True):
        super().__init__()
        mid = max(out_ch, int(in_ch * exp))
        self.expand = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.dw = nn.Conv2d(mid, mid, 3, stride=1, padding=1, groups=mid, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.proj = nn.Conv2d(mid, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.se = TinySE(out_ch, reduction=16) if se else nn.Identity()
        self.use_res = in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.expand(x)))
        y = self.act(self.bn2(self.dw(y)))
        y = self.bn3(self.proj(y))
        y = self.se(y)
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


# ---------------------------
# Global attention (light)
# ---------------------------


class GlobalAttentionLite(nn.Module):
    """
    Low-rank global self-attention over space with *downsampled* K/V for efficiency.
    - Q from x (HW tokens)
    - K,V from pooled x (H'W' tokens), H'W' << HW
    Complexity ~ O(HW * H'W'), not O((HW)^2).
    """

    def __init__(self, ch: int, attn_ch: Optional[int] = None, v_ch: Optional[int] = None, kv_down: int = 4):
        """
        ch:     input/output channels
        attn_ch: query/key dim (defaults ch//4)
        v_ch:   value dim before projection (defaults ch//2)
        kv_down: spatial downsample factor for K,V (avg-pool factor)
        """
        super().__init__()
        attn_ch = attn_ch or max(32, ch // 4)
        v_ch = v_ch or max(64, ch // 2)
        self.kv_down = kv_down

        self.q = nn.Conv2d(ch, attn_ch, 1, bias=False)
        self.k = nn.Conv2d(ch, attn_ch, 1, bias=False)
        self.v = nn.Conv2d(ch, v_ch, 1, bias=False)

        self.proj = nn.Conv2d(v_ch, ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.SiLU(inplace=True)

        self.scale = attn_ch**-0.5  # 1/sqrt(d)

        # Tiny local mixing after attention (DW3 + PW) to stabilize
        self.mix_dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.mix_pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.mix_bn = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.q(x)  # [B, aq, H, W]
        x_kv = F.avg_pool2d(x, kernel_size=self.kv_down) if self.kv_down > 1 else x
        k = self.k(x_kv)  # [B, aq, h', w']
        v = self.v(x_kv)  # [B, av, h', w']

        # flatten
        N = h * w
        h2, w2 = k.shape[-2:]
        M = h2 * w2
        qf = q.view(b, -1, N)  # [B, aq, N]
        kf = k.view(b, -1, M)  # [B, aq, M]
        vf = v.view(b, -1, M)  # [B, av, M]

        attn = torch.bmm(qf.transpose(1, 2), kf) * self.scale  # [B, N, M]
        attn = attn.softmax(dim=-1)

        y = torch.bmm(attn, vf.transpose(1, 2))  # [B, N, av]
        y = y.transpose(1, 2).contiguous().view(b, vf.size(1), h, w)  # [B, av, H, W]
        y = self.proj(y)  # [B, C, H, W]
        y = self.act(self.bn(y))

        # light local mixing + residual
        z = self.mix_pw(self.mix_dw(y))
        z = self.mix_bn(z)
        return x + self.act(z)


class AttnContextStack(nn.Module):
    """N layers of GlobalAttentionLite with residual stacking."""

    def __init__(
        self, ch: int, n_layers: int = 3, attn_ch: Optional[int] = None, v_ch: Optional[int] = None, kv_down: int = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            GlobalAttentionLite(ch, attn_ch=attn_ch, v_ch=v_ch, kv_down=kv_down) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.layers:
            x = blk(x)
        return x


# ---------------------------
# FPN-style top-down decoder
# ---------------------------


class FPNRefine(nn.Module):
    """Depthwise 3x3 + pointwise refine after top-down merge."""

    def __init__(self, ch: int):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw(self.dw(x))
        return self.act(self.bn(x))


# ---------------------------
# Model
# ---------------------------


@dataclass
class IRDWAttnFPNConfig:
    in_channels: int = 3
    num_keypoints: int = 15
    # encoder channel plan (C1, C2, C3) and depths per stage
    channels: Tuple[int, int, int] = (32, 64, 128)
    depths: Tuple[int, int, int] = (3, 4, 6)
    exp: int = 3
    se: bool = True

    # attention context
    attn_layers: int = 3
    attn_kv_down: int = 4  # 2~8; trade RF vs. cost
    attn_ch: Optional[int] = None
    attn_v_ch: Optional[int] = None

    # FPN width
    fpn_width: int = 96

    # outputs
    out_stride: int = 4  # 4 or 2
    use_offset_head: bool = False
    offset_per_kpt: int = 2
    deep_supervision: bool = False  # add an OS=8 auxiliary head


class IRDWAttnFPN(nn.Module):
    """
    Encoder: IRDW-only, deeper per stage
    Context: stacked global attention (downsampled K/V)
    Decoder: FPN (top-down) with lightweight refinement
    """

    def __init__(self, cfg: IRDWAttnFPNConfig):
        super().__init__()
        self.cfg = cfg
        C1, C2, C3 = cfg.channels
        D1, D2, D3 = cfg.depths

        # Stem -> Enc1 (OS=2)
        self.stem_dw = nn.Conv2d(
            cfg.in_channels, cfg.in_channels, 3, stride=2, padding=1, groups=cfg.in_channels, bias=False
        )
        self.stem_pw = nn.Conv2d(cfg.in_channels, C1, 1, bias=False)
        self.stem_bn = nn.BatchNorm2d(C1)
        self.stem_act = nn.SiLU(inplace=True)

        self.enc1 = nn.Sequential(*[IRDW(C1 if i == 0 else C1, C1, exp=cfg.exp, se=cfg.se) for i in range(D1)])

        # Down2 -> Enc2 (OS=4)
        self.down2 = DownDW(C1, C2)
        self.enc2 = nn.Sequential(*[IRDW(C2 if i == 0 else C2, C2, exp=cfg.exp, se=cfg.se) for i in range(D2)])

        # Down3 -> Enc3 (OS=8)
        self.down3 = DownDW(C2, C3)
        self.enc3 = nn.Sequential(*[IRDW(C3 if i == 0 else C3, C3, exp=cfg.exp, se=cfg.se) for i in range(D3)])

        # Context (OS=8)
        self.context = AttnContextStack(
            ch=C3, n_layers=cfg.attn_layers, attn_ch=cfg.attn_ch, v_ch=cfg.attn_v_ch, kv_down=cfg.attn_kv_down
        )

        # FPN lateral 1x1 to fpn_width
        W = cfg.fpn_width
        self.lat3 = nn.Conv2d(C3, W, 1, bias=False)  # from context output
        self.lat2 = nn.Conv2d(C2, W, 1, bias=False)
        self.lat1 = nn.Conv2d(C1, W, 1, bias=False)

        # top-down merges + refinement
        self.ref3 = FPNRefine(W)  # refine P3
        self.ref2 = FPNRefine(W)  # refine P2
        self.ref1 = FPNRefine(W)  # refine P1 (if OS=2)

        # Heads
        K = cfg.num_keypoints
        self.head_p2 = nn.Conv2d(W, K, 1)  # main @ OS=4
        if cfg.use_offset_head:
            self.off_p2 = nn.Conv2d(W, K * cfg.offset_per_kpt, 1)

        if cfg.out_stride == 2:
            self.head_p1 = nn.Conv2d(W, K, 1)
            if cfg.use_offset_head:
                self.off_p1 = nn.Conv2d(W, K * cfg.offset_per_kpt, 1)

        if cfg.deep_supervision:
            self.head_os8 = nn.Conv2d(C3, K, 1)

        assert cfg.out_stride in (4, 2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        # Encoder
        x = self.stem_act(self.stem_bn(self.stem_pw(self.stem_dw(x))))  # OS=2
        e1 = self.enc1(x)  # C1

        x = self.down2(e1)  # OS=4
        e2 = self.enc2(x)  # C2

        x = self.down3(e2)  # OS=8
        e3 = self.enc3(x)  # C3

        # Context @ OS=8
        c3 = self.context(e3)

        out: Dict[str, torch.Tensor | Dict[str, torch.Tensor]] = {}
        if self.cfg.deep_supervision:
            out["aux"] = {"os8": self.head_os8(c3)}

        # FPN top-down
        p3 = self.lat3(c3)  # OS=8
        p3 = self.ref3(p3)

        p2 = self.lat2(e2) + F.interpolate(p3, scale_factor=2, mode="nearest")  # OS=4
        p2 = self.ref2(p2)

        # main head @ OS=4
        heat = self.head_p2(p2)
        out["heatmap"] = torch.sigmoid(heat)
        if self.cfg.use_offset_head:
            out["offset"] = self.off_p2(p2)

        # Optional OS=2
        if self.cfg.out_stride == 2:
            p1 = self.lat1(e1) + F.interpolate(p2, scale_factor=2, mode="nearest")  # OS=2
            p1 = self.ref1(p1)
            heat2 = self.head_p1(p1)
            out["heatmap"] = torch.sigmoid(heat2)
            if self.cfg.use_offset_head:
                out["offset"] = self.off_p1(p1)

        return out


# ---------------------------
# Builder & quick test
# ---------------------------


def build_irdw_attn_fpn(
    num_keypoints: int = 15,
    variant: str = "small",  # "nano" | "small" | "base"
    depths: Tuple[int, int, int] = (3, 4, 6),
    out_stride: int = 4,
    attn_layers: int = 3,
    attn_kv_down: int = 4,
    use_offset_head: bool = False,
    deep_supervision: bool = False,
    fpn_width: int = 96,
    se: bool = True,
    exp: int = 3,
) -> IRDWAttnFPN:
    presets = {
        "nano": (24, 48, 96),
        "small": (32, 64, 128),
        "base": (48, 96, 192),
    }
    channels = presets.get(variant.lower())
    if channels is None:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(presets.keys())}.")

    cfg = IRDWAttnFPNConfig(
        in_channels=3,
        num_keypoints=num_keypoints,
        channels=channels,
        depths=depths,
        exp=exp,
        se=se,
        attn_layers=attn_layers,
        attn_kv_down=attn_kv_down,
        fpn_width=fpn_width,
        out_stride=out_stride,
        use_offset_head=use_offset_head,
        deep_supervision=deep_supervision,
    )
    return IRDWAttnFPN(cfg)


if __name__ == "__main__":
    B, C, H, W = 2, 3, 512, 288
    x = torch.randn(B, C, H, W)

    # Main: OS=4 heatmap
    model = build_irdw_attn_fpn(
        num_keypoints=15,
        variant="small",
        depths=(4, 5, 7),  # deeper IRDW per your request
        out_stride=4,
        attn_layers=3,
        attn_kv_down=4,
        use_offset_head=True,
        deep_supervision=True,
        fpn_width=96,
        se=True,
        exp=3,
    )
    with torch.no_grad():
        y = model(x)
    print("OS=4 heatmap:", y["heatmap"].shape)  # [B,15,H/4,W/4]
    if "offset" in y:
        print("OS=4 offset:", y["offset"].shape)
    if "aux" in y:
        print("Aux OS=8:", y["aux"]["os8"].shape)

    # Optional: OS=2
    model2 = build_irdw_attn_fpn(
        num_keypoints=15,
        variant="small",
        depths=(3, 4, 6),
        out_stride=2,
        attn_layers=2,
        attn_kv_down=4,
        use_offset_head=False,
        deep_supervision=False,
        fpn_width=80,
    )
    with torch.no_grad():
        y2 = model2(x)
    print("OS=2 heatmap:", y2["heatmap"].shape)  # [B,15,H/2,W/2]
