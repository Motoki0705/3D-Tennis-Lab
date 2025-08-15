# filename: development/court_pose/01_vit_heatmap/tests/conftest.py
import json
from pathlib import Path

import pytest
from dotmap import DotMap
from PIL import Image


@pytest.fixture(scope="session")
def project_root():
    """プロジェクトのルートディレクトリを返す"""
    return Path(__file__).parent.parent.parent.parent


@pytest.fixture
def dummy_config():
    """テスト用のダミー設定を返す"""
    config = {
        "model": {
            "vit_name": "vit_tiny_patch16_224",  # テストなので小さいモデルを使用
            "pretrained": False,
            "decoder_channels": [128, 64],
            "heatmap_channels": 15,
            "output_size": [56, 56],
        },
        "dataset": {
            "img_size": [224, 224],
            "heatmap_size": [56, 56],
            "heatmap_sigma": 2.0,
            "batch_size": 32,
            "num_workers": 8,
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "pin_memory": True,
            "persistent_workers": True,
        },
        "training": {
            "lr": 1e-4,
            "weight_decay": 1e-5,
        },
        "evaluation": {
            "pck_threshold": 0.05,
        },
    }
    return DotMap(config)


@pytest.fixture
def dummy_dataset_path(tmp_path):
    """
    一時ディレクトリにダミーの画像とアノテーションファイルを作成し、
    そのパスを返すfixture。
    """
    # 一時ディレクトリの構造
    data_dir = tmp_path / "data"
    img_dir = data_dir / "images"
    img_dir.mkdir(parents=True)
    annotation_path = data_dir / "annotation.json"

    # ダミー画像の作成 (2枚)
    dummy_image = Image.new("RGB", (640, 480), color="red")
    dummy_image.save(img_dir / "img1.png")
    dummy_image.save(img_dir / "img2.png")

    # ダミーアノテーションの作成
    annotations = {
        "images": [
            {"id": 1, "file_name": "img1.png", "width": 640, "height": 480},
            {"id": 2, "file_name": "img2.png", "width": 640, "height": 480},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                # kp0: visible, kp1: occluded, kp2: not labeled
                "keypoints": [100, 150, 2, 200, 250, 1, 0, 0, 0] + [0, 0, 0] * 12,
                "num_keypoints": 2,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "keypoints": [0, 0, 0] * 15,
                "num_keypoints": 0,
            },
        ],
        "categories": [{"id": 1, "name": "court", "keypoints": [f"kp_{i}" for i in range(15)], "skeleton": []}],
    }
    with open(annotation_path, "w") as f:
        json.dump(annotations, f)

    return img_dir, annotation_path
