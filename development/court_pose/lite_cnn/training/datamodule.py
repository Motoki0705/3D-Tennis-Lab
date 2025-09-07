from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple


from ...utils.lightning.base_datamodule import BaseDataModule
from .dataset import CourtKeypointDataset
from .transforms import get_train_transforms, get_val_transforms


@dataclass
class AugmentationConfig:
    p_horizontal_flip: float = 0.5
    p_affine: float = 0.3
    scale_limit: float = 0.05
    shift_limit: float = 0.05
    p_blur: float = 0.05
    blur_limit: int = 3
    p_color_jitter: float = 0.3
    brightness_limit: float = 0.1
    contrast_limit: float = 0.1
    saturation_limit: float = 0.05
    hue_limit: float = 0.03


@dataclass
class DataConfig:
    img_dir: str
    annotation_file: str
    img_size: Tuple[int, int] = (288, 512)  # (H, W)
    output_stride: int = 4
    sigma: float = 2.0
    use_offset: bool = False
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    # test_ratio is inferred as 1 - train_ratio - val_ratio
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


class CourtDataModule(BaseDataModule):
    def __init__(self, cfg: DataConfig, *, deep_supervision: bool = False, num_keypoints: Optional[int] = None):
        self.cfg = cfg
        height, width = cfg.img_size
        train_tf = get_train_transforms(height=height, width=width, **cfg.augmentation.__dict__)
        val_tf = get_val_transforms(height=height, width=width)

        dataset = CourtKeypointDataset(
            img_dir=Path(cfg.img_dir),
            annotation_file=Path(cfg.annotation_file),
            img_size=cfg.img_size,
            output_stride=cfg.output_stride,
            sigma=cfg.sigma,
            deep_supervision=deep_supervision,
            use_offset=cfg.use_offset,
            transform=None,  # will be set by BaseDataModule per split
            num_keypoints=num_keypoints,
        )

        super().__init__(
            config=type("_CfgProxy", (), {"dataset": cfg})(),
            dataset=dataset,
            train_transforms=train_tf,
            val_transforms=val_tf,
            test_transforms=val_tf,
        )
