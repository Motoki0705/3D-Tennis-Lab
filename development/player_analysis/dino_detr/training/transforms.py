import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    height: int,
    width: int,
    p_horizontal_flip: float = 0.5,
    p_affine: float = 0.4,
    scale_limit: float = 0.1,
    shift_limit: float = 0.1,
    p_blur: float = 0.1,
    blur_limit: int = 3,
    p_color_jitter: float = 0.2,
    brightness_limit: float = 0.1,
    contrast_limit: float = 0.1,
    saturation_limit: float = 0.05,
    hue_limit: float = 0.03,
):
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
            A.PadIfNeeded(min_height=height, min_width=width, border_mode=0, value=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], min_area=1.0, min_visibility=0.0),
    )


def get_val_transforms(height: int, width: int):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max(height, width), interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=height, min_width=width, border_mode=0, value=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], min_area=1.0, min_visibility=0.0),
    )
