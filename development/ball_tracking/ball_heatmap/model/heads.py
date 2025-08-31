import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class OffsetHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MultiScaleHeads(nn.Module):
    """
    Build heatmap/offset predictions at multiple output strides by pooling the shared features.
    Outputs are ordered from low->high resolution (i.e., larger stride -> smaller map first) to
    match dataset target ordering.
    """

    def __init__(self, in_channels: int, strides: list[int], heatmap_channels: int = 1, offset_channels: int = 2):
        super().__init__()
        # We assume the input feature map is at stride=1 relative to input image size.
        self.strides = list(strides)
        self.hmap_heads = nn.ModuleList([HeatmapHead(in_channels, heatmap_channels) for _ in self.strides])
        self.off_heads = nn.ModuleList([OffsetHead(in_channels, offset_channels) for _ in self.strides])

    def forward(self, x: torch.Tensor):
        heatmaps = []
        offsets = []
        _b, _c, _H, _W = x.shape
        for i, s in enumerate(self.strides):
            if s == 1:
                feat_s = x
            else:
                # Average pool to reduce to target stride
                feat_s = F.avg_pool2d(x, kernel_size=s, stride=s, ceil_mode=False)
            hmap = self.hmap_heads[i](feat_s)
            offs = self.off_heads[i](feat_s)
            heatmaps.append(hmap)
            offsets.append(offs)
        return heatmaps, offsets
