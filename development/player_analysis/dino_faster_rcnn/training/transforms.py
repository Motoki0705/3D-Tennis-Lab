from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _compute_target_dims(long_edge: int, aspect_ratio: Optional[float]) -> tuple[int, int]:
    """Compute (width, height) given a desired long edge and aspect ratio (w/h).

    - If aspect_ratio is None or invalid, returns a square (long_edge, long_edge).
    - Otherwise ensures max(width, height) == long_edge and width/height == aspect_ratio.
    """
    if aspect_ratio is None:
        return long_edge, long_edge
    try:
        ar = float(aspect_ratio)
    except Exception:
        return long_edge, long_edge
    if ar <= 0:
        return long_edge, long_edge
    if ar >= 1.0:
        w = int(long_edge)
        h = max(1, int(round(w / ar)))
    else:
        h = int(long_edge)
        w = max(1, int(round(h * ar)))
    return w, h


def get_train_transforms(image_size: int = 1024, aspect_ratio: Optional[float] = None) -> A.BasicTransform:
    target_w, target_h = _compute_target_dims(image_size, aspect_ratio)
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=target_h, min_width=target_w, border_mode=0, value=0),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.3),
            # Ensure float32 in [0, 1] before converting to tensor
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"], min_area=1.0, min_visibility=0.0),
    )


def get_val_transforms(image_size: int = 1024, aspect_ratio: Optional[float] = None) -> A.BasicTransform:
    target_w, target_h = _compute_target_dims(image_size, aspect_ratio)
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=target_h, min_width=target_w, border_mode=0, value=0),
            # Ensure float32 in [0, 1] before converting to tensor
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"], min_area=1.0, min_visibility=0.0),
    )


def get_infer_transforms(image_size: int = 1024, aspect_ratio: Optional[float] = None) -> A.BasicTransform:
    """Transforms for inference (no bbox params required).

    Resizes by long edge and pads to a fixed canvas that matches the desired aspect ratio,
    then converts to tensor in [0, 1].
    """
    target_w, target_h = _compute_target_dims(image_size, aspect_ratio)
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=target_h, min_width=target_w, border_mode=0, value=0),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ])


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

    def __init__(self, low_size: int, high_size: int, p_high: float = 0.0, aspect_ratio: Optional[float] = None):
        assert low_size > 0 and high_size > 0, "image sizes must be positive"
        self.low_size = int(low_size)
        self.high_size = int(high_size)
        self._p_high = float(max(0.0, min(1.0, p_high)))
        self.low_t = get_train_transforms(self.low_size, aspect_ratio=aspect_ratio)
        self.high_t = get_train_transforms(self.high_size, aspect_ratio=aspect_ratio)

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
