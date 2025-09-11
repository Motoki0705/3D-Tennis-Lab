from __future__ import annotations

import random
from dataclasses import dataclass

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


@dataclass
class ResolutionMixConfig:
    low_size: int
    high_size: int
    p_high: float = 0.0  # probability to apply high-res transform


class ResolutionMixTransform:
    """
    Wrap two albumentations transforms with different image sizes and mix them
    according to a controllable probability `p_high`.

    - Use `set_p_high(x)` to update the mix ratio during training.
    - Implements the same call signature as the underlying albumentations Compose.
    """

    def __init__(self, low_size: int, high_size: int, p_high: float = 0.0):
        assert low_size > 0 and high_size > 0, "image sizes must be positive"
        self.low_size = int(low_size)
        self.high_size = int(high_size)
        self._p_high = float(max(0.0, min(1.0, p_high)))
        self.low_t = get_train_transforms(self.low_size)
        self.high_t = get_train_transforms(self.high_size)

    def set_p_high(self, p: float) -> None:
        self._p_high = float(max(0.0, min(1.0, p)))

    @property
    def p_high(self) -> float:
        return self._p_high

    def __call__(self, *args, **kwargs):  # image=..., bboxes=..., class_labels=...
        use_high = random.random() < self._p_high
        if use_high:
            return self.high_t(*args, **kwargs)
        else:
            return self.low_t(*args, **kwargs)

    def __repr__(self) -> str:
        return f"ResolutionMixTransform(low={self.low_size}, high={self.high_size}, p_high={self._p_high:.2f})"
