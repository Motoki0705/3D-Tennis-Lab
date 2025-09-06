import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    height: int,
    width: int,
    p_horizontal_flip: float = 0.5,
    p_affine: float = 0.5,
    scale_limit: float = 0.1,
    shift_limit: float = 0.1,
    p_blur: float = 0.3,
    blur_limit: int = 5,
    p_color_jitter: float = 0.5,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    saturation_limit: float = 0.2,
    hue_limit: float = 0.1,
):
    """
    Returns a composition of transforms for training.
    These are designed to be applied to numpy arrays (H, W, C).
    The dataset is responsible for applying these transforms consistently
    across a sequence of frames (e.g., using ReplayCompose logic).
    """
    return A.Compose(
        [
            # Geometric transforms
            A.HorizontalFlip(p=p_horizontal_flip),
            A.Affine(
                scale=(1 - scale_limit, 1 + scale_limit),
                translate_percent=(-shift_limit, shift_limit),
                p=p_affine,
                keep_ratio=False,
            ),
            # Pixel-level transforms
            A.GaussianBlur(blur_limit=blur_limit, p=p_blur),
            A.ColorJitter(
                brightness=brightness_limit,
                contrast=contrast_limit,
                saturation=saturation_limit,
                hue=hue_limit,
                p=p_color_jitter,
            ),
            # Resize with aspect ratio preservation and padding (like letterbox)
            A.LongestMaxSize(max_size=max(height, width)),
            A.PadIfNeeded(min_height=height, min_width=width, border_mode=0, value=0),
            # Normalize and convert to tensor
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def get_val_transforms(height: int, width: int):
    """
    Returns a composition of transforms for validation/testing.
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max(height, width)),
            A.PadIfNeeded(min_height=height, min_width=width, border_mode=0, value=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def get_test_transforms(height: int, width: int):
    """
    Returns a composition of transforms for testing.
    """
    return get_val_transforms(height, width)
