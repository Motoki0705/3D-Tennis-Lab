# filename: development/court_pose/01_vit_heatmap/model_components/decoder.py

import torch.nn as nn


class UpsamplingDecoder(nn.Module):
    """
    転置畳み込み(ConvTranspose2d)を用いて、特徴量マップをアップサンプリングするデコーダ。
    各アップサンプリングブロックは、ConvTranspose2dの後にConv2dを追加してより深い構造になっています。

    ViTエンコーダからの出力を受け取り、ヒートマップヘッドに入力できる解像度まで拡大します。
    """

    def __init__(self, in_channels: int, decoder_channels: list[int]):
        """
        デコーダの初期化を行います。

        Args:
            in_channels (int): 入力特徴量マップのチャンネル数(例: ViT-Baseなら768)。
            decoder_channels (list[int]): デコーダ層のチャンネル数のリスト(例: [256, 128, 64])。
                                         リストの各要素が1つのアップサンプリングブロックの出力チャンネル数になります。
        """
        super().__init__()

        # 入力チャンネル数と設定ファイルからのチャンネルリストを結合
        # 例: in_channels=768, decoder_channels=[512, 256, 128, 64] -> channels=[768, 512, 256, 128, 64]
        channels = [in_channels, *decoder_channels]

        layers = []
        # チャンネルリストを元に、アップサンプリングブロックを動的に構築
        # (768 -> 512), (512 -> 256), (256 -> 128), (128 -> 64) のように層が作られる
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]

            # --- Upsampling Block ---
            # 1. 転置畳み込みで解像度を2倍にする
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,  # BatchNormを使うのでbiasは不要
                )
            )
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))

            # 2. 畳み込み層を追加してデコーダーを深くする
            layers.append(
                nn.Conv2d(
                    in_channels=out_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,  # BatchNormを使うのでbiasは不要
                )
            )
            layers.append(nn.BatchNorm2d(out_ch))
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
