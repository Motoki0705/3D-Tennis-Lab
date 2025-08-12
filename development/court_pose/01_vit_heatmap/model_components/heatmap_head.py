# filename: development/court_pose/01_vit_heatmap/model_components/heatmap_head.py
from torch import nn

class HeatmapHead(nn.Module):
    """
    最終的なヒートマップを出力するヘッド。
    """
    def __init__(self, in_channels, num_keypoints):
        super().__init__()
        self.head = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_keypoints,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
    def forward(self, x):
        return self.head(x)