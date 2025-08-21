import torch.nn as nn

from .decoder.context_pyramid_decoder import ContextPyramidDecoder
from .decoder.pixel_shuffle_attention_decoder import PixelShuffleAttentionDecoder
from .decoder.simple_decoder import UpsamplingDecoder
from .heatmap_head import HeatmapHead
from .vit_encoder import VitEncoder


class VitHeatmapModel(nn.Module):
    def __init__(
        self,
        vit_name="vit_base_patch16_224",
        pretrained=True,
        decoder_name="simple",
        decoder_channels=None,
        output_size=(224, 224),
        heatmap_channels=15,
    ):
        super().__init__()
        if decoder_channels is None:
            decoder_channels = [512, 256, 128, 64]
        self.encoder = VitEncoder(vit_name=vit_name, pretrained=pretrained)

        if decoder_name == "simple":
            self.decoder = UpsamplingDecoder(
                in_channels=self.encoder.embed_dim,
                decoder_channels=decoder_channels,
            )
        elif decoder_name == "pixel_shuffle_attention":
            self.decoder = PixelShuffleAttentionDecoder(
                in_channels=self.encoder.embed_dim, decoder_channels=decoder_channels, use_gn=False
            )
        elif decoder_name == "context_pyramid":
            self.decoder = ContextPyramidDecoder(
                in_channels=self.encoder.embed_dim,
                decoder_channels=decoder_channels,
                use_gn=False,
            )
        else:
            error_msg = f"Unknown decoder name: {decoder_name}"
            raise ValueError(error_msg)

        self.head = HeatmapHead(
            in_channels=decoder_channels[-1],
            num_keypoints=heatmap_channels,
        )

    def forward(self, x):
        features = self.encoder(x)
        upsampled = self.decoder(features)
        heatmaps = self.head(upsampled)
        return heatmaps
