import torch
import torch.nn as nn

from .decoder import UpsamplingDecoder
from .heads import MultiScaleHeads
from .vit_encoder import VitEncoder


class BallHeatmapModel(nn.Module):
    def __init__(
        self,
        vit_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        decoder_channels: list[int] | None = None,
        deep_supervision_strides: list[int] | None = None,
        heatmap_channels: int = 1,
        offset_channels: int = 2,
    ):
        super().__init__()
        if deep_supervision_strides is None:
            deep_supervision_strides = [8, 4]

        self.encoder = VitEncoder(vit_name=vit_name, pretrained=pretrained)
        self.decoder = UpsamplingDecoder(in_channels=self.encoder.embed_dim, decoder_channels=decoder_channels)
        self.heads = MultiScaleHeads(
            in_channels=(decoder_channels[-1] if decoder_channels else 64),
            strides=deep_supervision_strides,
            heatmap_channels=heatmap_channels,
            offset_channels=offset_channels,
        )

    def forward(self, x: torch.Tensor):
        feat = self.encoder(x)
        up = self.decoder(feat)
        heatmaps, offsets = self.heads(up)
        return {"heatmaps": heatmaps, "offsets": offsets}
