# filename: development/court_pose/01_vit_heatmap/model_components/decoder_ps.py
import torch
import torch.nn as nn


def _norm(ch, use_gn):
    return nn.GroupNorm(32, ch) if use_gn else nn.BatchNorm2d(ch)


class CBAM(nn.Module):
    def __init__(self, ch, r=16, k=7, use_gn=False):
        super().__init__()
        # Channel attention
        self.mlp = nn.Sequential(nn.Conv2d(ch, ch // r, 1), nn.ReLU(inplace=True), nn.Conv2d(ch // r, ch, 1))
        # Spatial attention
        self.conv_sp = nn.Conv2d(2, 1, k, padding=k // 2)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.max(x, dim=1, keepdim=True)[0]
        ca = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        x = x * ca
        sa = torch.sigmoid(self.conv_sp(torch.cat([avg, mx], dim=1)))
        return x * sa


class PixelShuffleUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_gn=False):
        super().__init__()
        self.expand = nn.Conv2d(in_ch, out_ch * 4, 1, bias=False)
        self.ps = nn.PixelShuffle(2)
        self.dw = nn.Conv2d(out_ch, out_ch, 7, padding=3, groups=out_ch, bias=False)
        self.pw = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.bn = _norm(out_ch, use_gn)
        self.act = nn.GELU()
        self.cbam = CBAM(out_ch)

    def forward(self, x):
        x = self.expand(x)
        x = self.ps(x)
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.cbam(x)
        return x


class PixelShuffleAttentionDecoder(nn.Module):
    """
    - 各段: 1x1でch拡張 → PixelShuffleでx2 → DWConv + PWConv → CBAM
    """

    def __init__(self, in_channels: int, decoder_channels: list[int], use_gn: bool = False):
        super().__init__()
        blocks = []
        c = in_channels
        for c_next in decoder_channels:
            blocks.append(PixelShuffleUpBlock(c, c_next, use_gn=use_gn))
            c = c_next
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)
