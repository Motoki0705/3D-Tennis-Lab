# filename: development/court_pose/01_vit_heatmap/model_components/decoder_context.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(ch, use_gn):
    return nn.GroupNorm(32, ch) if use_gn else nn.BatchNorm2d(ch)


class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch // r, 1)
        self.fc2 = nn.Conv2d(ch // r, ch, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class DWSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, d=1, use_gn=False):
        super().__init__()
        p = (k // 2) * d
        self.dw = nn.Conv2d(in_ch, in_ch, k, padding=p, dilation=d, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = _norm(out_ch, use_gn)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


class ASPP(nn.Module):
    def __init__(self, ch, out_ch, rates=(1, 3, 5), use_gn=False):
        super().__init__()
        self.br_1x1 = nn.Sequential(nn.Conv2d(ch, out_ch, 1, bias=False), _norm(out_ch, use_gn), nn.GELU())
        self.br_d = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, out_ch, 3, padding=r, dilation=r, bias=False),
                _norm(out_ch, use_gn),
                nn.GELU(),
            )
            for r in rates
        ])
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (1 + len(rates)), out_ch, 1, bias=False),
            _norm(out_ch, use_gn),
            nn.GELU(),
        )

    def forward(self, x):
        xs = [self.br_1x1(x)] + [b(x) for b in self.br_d]
        x = torch.cat(xs, dim=1)
        return self.project(x)


class NonLocal2D(nn.Module):
    def __init__(self, ch, reduction=2):
        super().__init__()
        c_int = max(32, ch // reduction)
        self.theta = nn.Conv2d(ch, c_int, 1, bias=False)
        self.phi = nn.Conv2d(ch, c_int, 1, bias=False)
        self.g = nn.Conv2d(ch, c_int, 1, bias=False)
        self.out = nn.Sequential(nn.Conv2d(c_int, ch, 1, bias=False), _norm(ch, use_gn=False))

    def forward(self, x):
        B, C, H, W = x.shape
        theta = self.theta(x).view(B, -1, H * W)  # B, C', HW
        phi = self.phi(x).view(B, -1, H * W)  # B, C', HW
        attn = torch.softmax(theta.transpose(1, 2) @ phi, dim=-1)  # B, HW, HW
        g = self.g(x).view(B, -1, H * W).transpose(1, 2)  # B, HW, C'
        y = (attn @ g).transpose(1, 2).view(B, -1, H, W)  # B, C', H, W
        y = self.out(y)
        return x + y


class ContextUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_gn=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = DWSeparableConv(in_ch, out_ch, k=7, use_gn=use_gn)
        self.aspp = ASPP(out_ch, out_ch, rates=(1, 3, 5), use_gn=use_gn)
        self.se = SEBlock(out_ch)
        self.conv2 = DWSeparableConv(out_ch, out_ch, k=3, use_gn=use_gn)

    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.aspp(x)
        x = self.se(x)
        x = self.conv2(x)
        return x


class ContextPyramidDecoder(nn.Module):
    """
    - 入口で Non-Local によるグローバル文脈注入
    - 各段: Resize-Conv + 大カーネル + ASPP + SE
    """

    def __init__(self, in_channels: int, decoder_channels: list[int], use_gn: bool = False):
        super().__init__()
        self.gc = NonLocal2D(in_channels)
        blocks = []
        c = in_channels
        for c_next in decoder_channels:
            blocks.append(ContextUpBlock(c, c_next, use_gn=use_gn))
            c = c_next
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.gc(x)
        return self.blocks(x)
