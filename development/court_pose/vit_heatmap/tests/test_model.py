# filename: development/court_pose/01_vit_heatmap/tests/test_model.py
import torch

from ..model.decoder import UpsamplingDecoder
from ..model.vit_encoder import VitEncoder
from ..model.vit_heatmap import VitHeatmapModel


def test_vit_encoder(dummy_config):
    """VitEncoderが正しい形状の特徴量マップを生成するかテスト"""
    cfg = dummy_config.model
    # NOTE: dummy_configは 'vit_tiny_patch16_224' を使用
    # embed_dim=192, num_heads=3, depth=12
    encoder = VitEncoder(vit_name=cfg.vit_name, pretrained=cfg.pretrained)
    assert encoder.embed_dim == 192

    # (B, C, H, W)
    dummy_input = torch.randn(2, 3, 224, 224)
    features = encoder(dummy_input)

    # 224x224, patch16 -> 14x14 patches
    # (B, D, H/P, W/P)
    assert features.shape == (2, 192, 14, 14)


def test_upsampling_decoder(dummy_config):
    """UpsamplingDecoderが正しく特徴量マップをアップサンプリングするかテスト"""
    cfg = dummy_config.model
    in_channels = 192  # from vit_tiny
    decoder = UpsamplingDecoder(in_channels=in_channels, decoder_channels=cfg.decoder_channels)

    # (B, D, H/P, W/P)
    dummy_input = torch.randn(2, in_channels, 14, 14)
    upsampled = decoder(dummy_input)

    # 2回のアップサンプリング (14x14 -> 28x28 -> 56x56)
    # (B, C_out, H_out, W_out)
    final_channels = cfg.decoder_channels[-1]
    assert upsampled.shape == (2, final_channels, 56, 56)


def test_vit_heatmap_model_e2e(dummy_config):
    """VitHeatmapModelのEnd-to-Endフォワードパスをテスト"""
    cfg = dummy_config.model
    model = VitHeatmapModel(
        vit_name=cfg.vit_name,
        pretrained=cfg.pretrained,
        decoder_channels=cfg.decoder_channels,
        output_size=cfg.output_size,
        heatmap_channels=cfg.heatmap_channels,
    )

    # (B, C, H, W)
    dummy_input = torch.randn(2, 3, 224, 224)
    heatmaps = model(dummy_input)

    # (B, K, H_out, W_out)
    assert heatmaps.shape == (2, cfg.heatmap_channels, cfg.output_size[0], cfg.output_size[1])
