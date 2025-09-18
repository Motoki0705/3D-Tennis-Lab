from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass

# --- Sub-modules ---


class ConvGRUCell(nn.Module):
    """A simple ConvGRU cell implementation."""

    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv_gates = nn.Conv2d(input_dim + hidden_dim, hidden_dim * 2, kernel_size, padding=padding)
        self.conv_can = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)

    def forward(self, input_tensor, h_cur):
        combined = torch.cat([input_tensor, h_cur], dim=1)
        gates = self.conv_gates(combined)
        reset_gate, update_gate = gates.chunk(2, 1)
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)

        combined_reset = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_can(combined_reset)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next


class DinoV3Encoder(nn.Module):
    """Encoder that uses a local DINOv3 ViT model to extract patch tokens."""

    def __init__(self, repo_dir, entry, weights, freeze=True):
        super().__init__()
        self.vit = torch.hub.load(repo_dir, entry, source="local", weights=weights)
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False
        self.patch_size = self.vit.patch_size
        self.embed_dim = self.vit.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.vit.forward_features(x)
            tokens = feats.get("x_norm_patchtokens")

        b, n, c = tokens.shape
        h = x.shape[-2] // self.patch_size
        w = x.shape[-1] // self.patch_size
        feat_2d = tokens.transpose(1, 2).contiguous().view(b, c, h, w)
        return feat_2d


class HeatmapDecoder(nn.Module):
    """Decodes features into a heatmap using transposed convolutions."""

    def __init__(self, input_dim, output_stride):
        super().__init__()
        self.layers = nn.ModuleList()
        num_upsamples = int(torch.log2(torch.tensor(output_stride)).item())
        current_dim = input_dim
        for i in range(num_upsamples):
            next_dim = max(current_dim // 2, 16)
            self.layers.append(nn.ConvTranspose2d(current_dim, next_dim, kernel_size=4, stride=2, padding=1))
            self.layers.append(nn.ReLU(inplace=True))
            current_dim = next_dim
        self.layers.append(nn.Conv2d(current_dim, 1, kernel_size=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# --- Main Network ---


@dataclass
class NetConfig:
    repo_dir: str = "third_party/dinov3"
    entry: str = "dinov3_vits16"
    weights: str = "third_party/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    encoder_out_dim: int = 384
    rnn_hidden_dim: int = 128
    output_stride: int = 4


class SequenceHeatmapNet(nn.Module):
    def __init__(self, cfg: NetConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = DinoV3Encoder(cfg.repo_dir, cfg.entry, cfg.weights)
        self.conv_gru = ConvGRUCell(cfg.encoder_out_dim, cfg.rnn_hidden_dim, kernel_size=3)
        self.decoder = HeatmapDecoder(cfg.rnn_hidden_dim, cfg.output_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        h_feat, w_feat = h // self.encoder.patch_size, w // self.encoder.patch_size

        h_gru = torch.zeros(b, self.cfg.rnn_hidden_dim, h_feat, w_feat, device=x.device)

        outputs = []
        for i in range(t):
            frame = x[:, i, :, :, :]
            frame_features = self.encoder(frame)
            h_gru = self.conv_gru(frame_features, h_gru)
            heatmap = self.decoder(h_gru)
            outputs.append(heatmap)

        return torch.stack(outputs, dim=1)


def build_model(model_cfg: dict) -> SequenceHeatmapNet:
    cfg = NetConfig(**model_cfg)
    return SequenceHeatmapNet(cfg)


__all__ = ["SequenceHeatmapNet", "build_model", "NetConfig"]
