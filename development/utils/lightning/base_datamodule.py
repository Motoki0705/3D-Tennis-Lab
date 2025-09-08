# filename: development/court_pose/01_vit_heatmap/datamodule.py
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
        dataset,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
    ):
        super().__init__()
        # Some callers pass Hydra DictConfig (supported), others pass lightweight objects.
        # Save hparams when supported; otherwise skip without failing.
        try:
            self.save_hyperparameters(config)
        except Exception:
            pass
        self.config = config
        self.full_dataset = dataset
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

    def setup(self, stage=None):
        n_data = len(self.full_dataset)
        n_train = int(n_data * self.config.dataset.train_ratio)
        n_val = int(n_data * self.config.dataset.val_ratio)
        n_test = n_data - n_train - n_val

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.full_dataset, [n_train, n_val, n_test]
        )

        # 各データセットに適切なTransformを適用
        if self.train_transforms:
            self.train_dataset.dataset.transform = self.train_transforms
        if self.val_transforms:
            self.val_dataset.dataset.transform = self.val_transforms
        if self.test_transforms:
            self.test_dataset.dataset.transform = self.test_transforms

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=True,
            num_workers=self.config.dataset.num_workers,
            pin_memory=self.config.dataset.pin_memory,
            persistent_workers=self.config.dataset.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            pin_memory=self.config.dataset.pin_memory,
            persistent_workers=self.config.dataset.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            pin_memory=self.config.dataset.pin_memory,
            persistent_workers=self.config.dataset.persistent_workers,
        )
