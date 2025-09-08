from pathlib import Path

from ...utils.lightning.base_datamodule import BaseDataModule
from ..vit_heatmap.dataset import CourtKeypointDataset


class CourtDataModule(BaseDataModule):
    def __init__(self, config, train_transforms=None, val_transforms=None, test_transforms=None):
        dataset = CourtKeypointDataset(
            img_dir=Path(config.dataset.img_dir),
            annotation_file=Path(config.dataset.annotation_file),
            img_size=config.dataset.img_size,
            heatmap_size=config.dataset.heatmap_size,
            sigma=config.dataset.heatmap_sigma,
            transform=None,
        )

        super().__init__(
            config=config,
            dataset=dataset,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
        )
