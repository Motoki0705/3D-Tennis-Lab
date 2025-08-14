import albumentations as A
from albumentations.pytorch import ToTensorV2


def prepare_transforms(img_size, heatmap_size, sigma):
    train_transforms = A.Compose(
        [
            A.Resize(img_size[0], img_size[1]),
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
    val_transforms = A.Compose(
        [
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
    return train_transforms, val_transforms
