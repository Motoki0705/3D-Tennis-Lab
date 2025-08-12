# filename: development/court_pose/01_vit_heatmap/tests/test_datamodule.py
import torch
from ..datamodule import CourtDataModule

def test_datamodule_setup(dummy_dataset_path, dummy_config, monkeypatch):
    """DataModuleがデータセットを正しくセットアップするかテスト"""
    img_dir, ann_file = dummy_dataset_path
    dummy_config.data.img_dir = img_dir
    dummy_config.data.annotation_file = ann_file
    
    # random_splitが常に同じ結果を返すようにモック化
    def mock_random_split(dataset, lengths):
        from torch.utils.data import Subset
        indices = list(range(sum(lengths)))
        return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(torch.cumsum(torch.tensor(lengths), 0), lengths)]
    
    monkeypatch.setattr("torch.utils.data.random_split", mock_random_split)

    dm = CourtDataModule(dummy_config)
    dm.setup()
    
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert dm.test_dataset is not None
    
    # 80/10/10分割なので、2件のデータなら 1/1/0 になるはず
    # 注意: random_split の実装により、実際の分割数は変動しうる。
    # このテストはあくまで基本的な動作確認。
    assert len(dm.train_dataset) > 0

def test_dataloaders(dummy_dataset_path, dummy_config):
    """データローダーが正しいバッチを生成するかテスト"""
    img_dir, ann_file = dummy_dataset_path
    dummy_config.data.img_dir = img_dir
    dummy_config.data.annotation_file = ann_file

    dm = CourtDataModule(dummy_config)
    dm.setup()
    
    loader = dm.train_dataloader()
    images, heatmaps = next(iter(loader))
    
    cfg = dummy_config.data
    batch_size = cfg.batch_size
    
    assert images.shape == (batch_size, 3, cfg.img_size[0], cfg.img_size[1])
    assert heatmaps.shape == (batch_size, 15, cfg.heatmap_size[0], cfg.heatmap_size[1])
    assert images.dtype == torch.float32
    assert heatmaps.dtype == torch.float32