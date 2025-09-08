import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


def _pad_if_needed(height: int, width: int, *, border_mode: int = cv2.BORDER_CONSTANT, value: int = 0):
    """Create PadIfNeeded compatible across albumentations versions.

    In some versions, the kwarg is `value`, in others it's `border_value`.
    """
    try:
        return A.PadIfNeeded(min_height=height, min_width=width, border_mode=border_mode, value=value)
    except TypeError:
        try:
            return A.PadIfNeeded(min_height=height, min_width=width, border_mode=border_mode, border_value=value)
        except TypeError:
            return A.PadIfNeeded(min_height=height, min_width=width, border_mode=border_mode)


def get_train_transforms(
    height: int,
    width: int,
    p_horizontal_flip: float = 0.5,
    p_affine: float = 0.3,
    scale_limit: float = 0.05,
    shift_limit: float = 0.05,
    p_blur: float = 0.05,
    blur_limit: int = 3,
    p_color_jitter: float = 0.3,
    brightness_limit: float = 0.1,
    contrast_limit: float = 0.1,
    saturation_limit: float = 0.05,
    hue_limit: float = 0.03,
):
    """Keypoint-aware training transforms (applied to single images).

    Use ReplayCompose externally if you need consistent params over sequences.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=p_horizontal_flip),
            A.Affine(
                scale=(1 - scale_limit, 1 + scale_limit),
                translate_percent=(-shift_limit, shift_limit),
                p=p_affine,
                keep_ratio=False,
            ),
            A.GaussianBlur(blur_limit=blur_limit, p=p_blur),
            A.ColorJitter(
                brightness=brightness_limit,
                contrast=contrast_limit,
                saturation=saturation_limit,
                hue=hue_limit,
                p=p_color_jitter,
            ),
            A.LongestMaxSize(max_size=max(height, width), interpolation=cv2.INTER_AREA),
            _pad_if_needed(height, width, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def get_val_transforms(height: int, width: int):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max(height, width), interpolation=cv2.INTER_AREA),
            _pad_if_needed(height, width, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def get_test_transforms(height: int, width: int):
    return get_val_transforms(height, width)
