import math

import timm
import torch.nn as nn


class VitEncoder(nn.Module):
    """
    ViT encoder using timm that outputs a 2D feature map (without CLS token).
    """

    def __init__(self, vit_name: str = "vit_base_patch16_224", pretrained: bool = True):
        super().__init__()
        self.vit = timm.create_model(vit_name, pretrained=pretrained)
        self.embed_dim = getattr(self.vit, "embed_dim", None) or getattr(self.vit, "num_features")

    def forward(self, x):
        features = self.vit.forward_features(x)  # [B, N+1, D]
        # remove CLS token if present
        if features.dim() == 3 and features.shape[1] > 1:
            features = features[:, 1:]

        b, n, d = features.shape
        h = w = int(math.sqrt(n))
        features = features.permute(0, 2, 1).reshape(b, d, h, w)
        return features
