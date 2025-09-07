# filename: development/court_pose/01_vit_heatmap/model_components/vit_encoder.py
import math

import timm
import torch.nn as nn


class VitEncoder(nn.Module):
    """
    timmライブラリからVision Transformerをロードし、
    2D特徴量マップを出力するエンコーダ。
    """

    def __init__(self, vit_name="vit_base_patch16_224", pretrained=True):
        super().__init__()
        self.vit = timm.create_model(vit_name, pretrained=pretrained)
        # ViTの出力次元を取得
        self.embed_dim = self.vit.embed_dim

    def forward(self, x):
        # ViTの特徴量抽出器を実行
        # [B, 3, H, W] -> [B, N, D] (N: num_patches + 1 for CLS token)
        features = self.vit.forward_features(x)

        # CLSトークンを除去
        # [B, 197, 768] -> [B, 196, 768] for 224x224 input
        features = features[:, 1:]

        b, n, d = features.shape
        # パッチグリッドのサイズを計算 (e.g., sqrt(196) = 14)
        h = w = int(math.sqrt(n))

        # [B, N, D] -> [B, D, H, W] の2D特徴量マップに変換
        features = features.permute(0, 2, 1).reshape(b, d, h, w)

        return features
