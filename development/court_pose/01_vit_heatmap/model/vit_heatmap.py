import torch.nn as nn

from .decoder import UpsamplingDecoder
from .heatmap_head import HeatmapHead
from .vit_encoder import VitEncoder


class VitHeatmapModel(nn.Module):
    def __init__(
        self,
        vit_name="vit_base_patch16_224",
        pretrained=True,
        decoder_channels=None,
        output_size=(224, 224),
        heatmap_channels=15,
    ):
        if decoder_channels is None:
            decoder_channels = [512, 256, 128, 64]
        self.encoder = VitEncoder(vit_name=vit_name, pretrained=pretrained)
        self.decoder = UpsamplingDecoder(
            in_channels=self.encoder.embed_dim, decoder_channels=decoder_channels, output_size=output_size
        )
        self.head = HeatmapHead(
            in_channels=decoder_channels[-1],
            num_keypoints=heatmap_channels,
        )

    def forward(self, x):
        features = self.encoder(x)
        upsampled = self.decoder(features)
        heatmaps = self.head(upsampled)
        return heatmaps
