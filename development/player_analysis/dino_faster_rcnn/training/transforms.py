from __future__ import annotations


import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 1024) -> A.BasicTransform:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=0),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.3),
            # Ensure float32 in [0, 1] before converting to tensor
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"], min_area=1.0, min_visibility=0.0),
    )


def get_val_transforms(image_size: int = 1024) -> A.BasicTransform:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=0),
            # Ensure float32 in [0, 1] before converting to tensor
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"], min_area=1.0, min_visibility=0.0),
    )
