# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DINOv3 backbone for a standard DETR model.
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding

# DINOv3モデルのリポジトリパスやモデル名を指定します。
# ローカルにdinov3リポジトリがない場合は、自動的にtorch.hubからダウンロードされます。
# Defaults are overridden by args.dino_repo_dir / args.dino_model_name
DINO_REPO_DIR_DEFAULT = "third_party/dinov3"
DINO_MODEL_NAME_DEFAULT = "dinov3_vitl16"  # 例: dinov3_vitl16 (Large)


class DinoBackbone(nn.Module):
    """
    DINOv3 Vision Transformer backbone for standard DETR.
    This backbone returns the features of the last layer of the ViT, reshaped into a 2D feature map.
    """

    def __init__(self, dino_model: nn.Module, train_backbone: bool):
        super().__init__()
        # 事前学習済みの重みをフリーズするかどうかを設定
        for name, parameter in dino_model.named_parameters():
            if not train_backbone:
                parameter.requires_grad_(False)

        self.body = dino_model
        self.num_channels = dino_model.embed_dim  # Transformerに渡すチャンネル数はViTの埋め込み次元

    def forward(self, tensor_list: NestedTensor):
        # DINOv3のget_intermediate_layersを使い、最終層の特徴量のみを取得
        # n=1 は最後の1ブロックを意味します。
        # reshape=True で (B, C, H, W) の2Dマップ形式に変換します。
        xs = self.body.get_intermediate_layers(tensor_list.tensors, n=1, reshape=True, return_class_token=False)[
            0
        ]  # get_intermediate_layersはタプルを返すため、最初の要素を取得

        # マスクを特徴マップのサイズに合わせる
        m = tensor_list.mask
        assert m is not None
        mask = F.interpolate(m[None].float(), size=xs.shape[-2:]).to(torch.bool)[0]

        # DETRの標準的なバックボーンは辞書形式で返すため、互換性のために合わせる
        out: Dict[str, NestedTensor] = {}
        out["0"] = NestedTensor(xs, mask)
        return out


class Joiner(nn.Sequential):
    """
    バックボーンと位置エンコーディングを結合するクラス。
    """

    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        # backbone (DinoBackbone) を実行
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        # DinoBackboneは{"0": NestedTensor}という辞書を返す
        for name, x in xs.items():
            out.append(x)
            # 位置エンコーディングを計算
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_dino_backbone(args):
    """
    DINOv3バックボーンを構築するメイン関数。
    """
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0

    # DINOv3モデルをロード（repoとモデル名は引数から可変）
    repo_dir = getattr(args, "dino_repo_dir", DINO_REPO_DIR_DEFAULT)
    model_name = getattr(args, "dino_model_name", DINO_MODEL_NAME_DEFAULT)
    # is_main_process()は分散学習時にメインプロセスのみが重みをダウンロード/ロードするようにします。
    try:
        dino_vit_model = torch.hub.load(repo_dir, model_name, source="local", pretrained=is_main_process())
    except (FileNotFoundError, IsADirectoryError):
        print(f"Could not find DINOv3 repository at '{repo_dir}'. Loading from torch.hub.")
        dino_vit_model = torch.hub.load("facebookresearch/dinov3", model_name, pretrained=is_main_process())

    # DinoBackboneをインスタンス化
    backbone = DinoBackbone(dino_vit_model, train_backbone)

    # Backboneと位置エンコーディングをJoinerで結合
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
