# filename: development/court_pose/01_vit_heatmap/model_components/decoder.py

import torch.nn as nn

class UpsamplingDecoder(nn.Module):
    """
    転置畳み込み（ConvTranspose2d）を用いて、特徴量マップをアップサンプリングするデコーダ。

    ViTエンコーダからの出力を受け取り、ヒートマップヘッドに入力できる解像度まで拡大します。
    """

    def __init__(self, in_channels: int, decoder_channels: list[int]):
        """
        デコーダの初期化を行います。

        Args:
            in_channels (int): 入力特徴量マップのチャンネル数（例: ViT-Baseなら768）。
            decoder_channels (list[int]): デコーダ層のチャンネル数のリスト（例: [256, 128, 64]）。
                                         リストの各要素が1つのアップサンプリングブロックの出力チャンネル数になります。
        """
        super().__init__()

        # 入力チャンネル数と設定ファイルからのチャンネルリストを結合
        # 例: in_channels=768, decoder_channels=[256, 128, 64] -> channels=[768, 256, 128, 64]
        channels = [in_channels] + decoder_channels

        layers = []
        # チャンネルリストを元に、アップサンプリングブロックを動的に構築
        # (768 -> 256), (256 -> 128), (128 -> 64) のように層が作られる
        for i in range(len(channels) - 1):
            # 転置畳み込みで解像度を2倍にする
            # kernel_size=4, stride=2, padding=1 は一般的なアップサンプリングの組み合わせ
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=channels[i],
                    out_channels=channels[i+1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False # BatchNormを使うのでbiasは不要
                )
            )
            # バッチ正規化で学習を安定させる
            layers.append(nn.BatchNorm2d(channels[i+1]))
            # ReLU活性化関数で非線形性を導入
            layers.append(nn.ReLU(inplace=True))

        # 構築した層をSequentialコンテナにまとめる
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        フォワードパス。

        Args:
            x (torch.Tensor): エンコーダからの入力特徴量マップ。
                              Shape: [B, in_channels, H, W]

        Returns:
            torch.Tensor: アップサンプリングされた特徴量マップ。
                          Shape: [B, decoder_channels[-1], H*2^N, W*2^N]
                          (Nはアップサンプリングブロックの数)
        """
        return self.decoder(x)