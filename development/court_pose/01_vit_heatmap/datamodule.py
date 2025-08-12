# filename: development/court_pose/01_vit_heatmap/datamodule.py
import albumentations as A
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split

from .dataset import CourtKeypointDataset


class CourtDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        # データ拡張パイプライン
        train_transforms = A.Compose([
            A.Resize(self.config.data.img_size[0], self.config.data.img_size[1]),
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        val_transforms = A.Compose([
            A.Resize(self.config.data.img_size[0], self.config.data.img_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        # データセットのインスタンス化
        full_dataset = CourtKeypointDataset(
            img_dir=self.config.data.img_dir,
            annotation_file=self.config.data.annotation_file,
            img_size=self.config.data.img_size,
            heatmap_size=self.config.data.heatmap_size,
            sigma=self.config.data.heatmap_sigma
        )

        # 80/10/10 に分割
        n_data = len(full_dataset)
        n_train = int(n_data * 0.8)
        n_val = int(n_data * 0.1)
        n_test = n_data - n_train - n_val
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [n_train, n_val, n_test]
        )
        
        # 各データセットに適切なTransformを適用
        self.train_dataset.dataset.transform = train_transforms
        self.val_dataset.dataset.transform = val_transforms
        self.test_dataset.dataset.transform = val_transforms


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
        )