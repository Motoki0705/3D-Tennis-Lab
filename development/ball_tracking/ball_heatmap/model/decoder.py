import torch
import torch.nn as nn


class UpsamplingDecoder(nn.Module):
    """
    Simple transposed-conv based decoder that upsamples ViT grid features to input resolution.
    """

    def __init__(self, in_channels: int, decoder_channels: list[int] | None = None):
        super().__init__()
        if decoder_channels is None:
            decoder_channels = [512, 256, 128, 64]
        channels = [in_channels, *decoder_channels]
        layers: list[nn.Module] = []
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
