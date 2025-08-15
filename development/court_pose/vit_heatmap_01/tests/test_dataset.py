# filename: development/court_pose/01_vit_heatmap/tests/test_dataset.py
import torch

from development.court_pose.vit_heatmap_01.dataset import CourtKeypointDataset


def test_dataset_initialization(dummy_dataset_path, dummy_config):
    """データセットが正しく初期化されるかテスト"""
    img_dir, ann_file = dummy_dataset_path
    dataset = CourtKeypointDataset(
        img_dir=img_dir,
        annotation_file=ann_file,
        img_size=dummy_config.dataset.img_size,
        heatmap_size=dummy_config.dataset.heatmap_size,
        sigma=dummy_config.dataset.heatmap_sigma,
    )
    assert len(dataset) == 2, "アノテーションの数が正しく読み込めていない"


def test_dataset_getitem(dummy_dataset_path, dummy_config):
    """__getitem__が正しい形状と型のテンソルを返すかテスト"""
    img_dir, ann_file = dummy_dataset_path
    cfg = dummy_config.dataset
    dataset = CourtKeypointDataset(
        img_dir=img_dir,
        annotation_file=ann_file,
        img_size=cfg.img_size,
        heatmap_size=cfg.heatmap_size,
        sigma=cfg.heatmap_sigma,
        transform=None,  # テストでは拡張なし
    )

    image, heatmap = dataset[0]  # 最初のデータを取得

    # 型の確認
    assert isinstance(image, torch.Tensor)
    assert isinstance(heatmap, torch.Tensor)

    # 形状の確認
    assert image.shape == (3, cfg.img_size[0], cfg.img_size[1])
    assert heatmap.shape == (15, cfg.heatmap_size[0], cfg.heatmap_size[1])

    # 値の範囲の確認
    assert image.min() >= 0.0 and image.max() <= 1.0
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0


def test_heatmap_generation(dummy_dataset_path, dummy_config):
    """ヒートマップがキーポイント位置にピークを持つかテスト"""
    img_dir, ann_file = dummy_dataset_path
    cfg = dummy_config.dataset
    dataset = CourtKeypointDataset(
        img_dir=img_dir,
        annotation_file=ann_file,
        img_size=cfg.img_size,
        heatmap_size=cfg.heatmap_size,
        sigma=cfg.heatmap_sigma,
    )

    _, heatmap = dataset[0]

    # keypoint 0 (visible) のヒートマップを確認
    heatmap_kp0 = heatmap[0]
    assert heatmap_kp0.max() > 0.9, "可視キーポイントのピークが低い"
    # 最大値の位置を取得
    peak_idx = torch.argmax(heatmap_kp0)
    peak_y = peak_idx // cfg.heatmap_size[1]
    peak_x = peak_idx % cfg.heatmap_size[1]

    # 元の座標 (100, 150) がリサイズ後のヒートマップ座標に正しくマッピングされているか
    expected_x = int(100 * (cfg.heatmap_size[1] / 640))
    expected_y = int(150 * (cfg.heatmap_size[0] / 480))
    assert abs(peak_x - expected_x) <= 1
    assert abs(peak_y - expected_y) <= 1

    # keypoint 2 (not labeled) のヒートマップを確認
    heatmap_kp2 = heatmap[2]
    assert torch.all(heatmap_kp2 == 0), "ラベルなしキーポイントのヒートマップが0でない"
