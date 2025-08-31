import json
from pathlib import Path

import cv2
import numpy as np
import torch


class IdentityTransform:
    def __call__(self, image, keypoints):
        import torch

        # No resize/flip; convert to tensor [C,H,W]
        img_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return {"image": img_t, "keypoints": keypoints}


def _make_toy_coco(tmp_path: Path):
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Create three 32x32 images
    paths = []
    fn1 = img_dir / "game1" / "v1" / "frame1.png"
    fn1.parent.mkdir(parents=True, exist_ok=True)
    img1 = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.circle(img1, (16, 16), 3, (255, 255, 255), -1)
    cv2.imwrite(str(fn1), cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
    paths.append((1, str(Path("images") / "game1" / "v1" / "frame1.png"), 32, 32))

    fn2 = img_dir / "game2" / "frame2.png"
    fn2.parent.mkdir(parents=True, exist_ok=True)
    img2 = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(fn2), cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
    paths.append((2, str(Path("images") / "game2" / "frame2.png"), 32, 32))

    fn3 = img_dir / "game3" / "v2" / "frame3.png"
    fn3.parent.mkdir(parents=True, exist_ok=True)
    img3 = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.circle(img3, (8, 24), 2, (255, 255, 255), -1)
    cv2.imwrite(str(fn3), cv2.cvtColor(img3, cv2.COLOR_RGB2BGR))
    paths.append((3, str(Path("images") / "game3" / "v2" / "frame3.png"), 32, 32))

    images = [{"id": img_id, "file_name": file_name, "width": w, "height": h} for (img_id, file_name, w, h) in paths]

    # Annotations: id, image_id, category_id(1 ball), bbox
    annotations = [
        {"id": 101, "image_id": 1, "category_id": 1, "bbox": [14.0, 14.0, 4.0, 4.0], "area": 16.0},
        # image 2: no ball (negative)
        {"id": 103, "image_id": 3, "category_id": 1, "bbox": [7.0, 23.0, 2.0, 2.0], "area": 4.0},
    ]

    categories = [
        {"id": 1, "name": "ball", "supercategory": "object", "keypoints": ["center"], "skeleton": []},
        {"id": 2, "name": "player"},
    ]

    coco = {"images": images, "annotations": annotations, "categories": categories}
    ann_path = tmp_path / "annotation.json"
    with open(ann_path, "w") as f:
        json.dump(coco, f)
    return img_dir, ann_path


def test_ball_dataset_multiscale_and_negatives(tmp_path):
    # Lazy import with albumentations stub to avoid dependency at test-time
    import sys, types, importlib

    try:
        importlib.import_module("albumentations")
    except Exception:
        sys.modules["albumentations"] = types.SimpleNamespace(
            Compose=lambda *a, **k: None, KeypointParams=lambda *a, **k: None
        )
    try:
        importlib.import_module("albumentations.pytorch")
    except Exception:
        sys.modules["albumentations.pytorch"] = types.SimpleNamespace(ToTensorV2=object)
    from development.ball_tracking.ball_heatmap.dataset import BallDataset, HeatmapSpec

    img_dir, ann_path = _make_toy_coco(tmp_path)

    specs = [HeatmapSpec(8, 1.0), HeatmapSpec(4, 1.0)]
    ds = BallDataset(
        img_dir=img_dir,
        annotation_file=ann_path,
        img_size=(32, 32),
        heatmap_specs=specs,
        negatives="use",
        version_field="view",
        transform=IdentityTransform(),
    )

    assert len(ds) == 3  # includes negative image
    assert set(ds.versions) == {"v1", "v2", "unknown"}

    # Find indices for pos and neg
    idx_pos = ds.versions.index("v1")
    sample_pos = ds[idx_pos]
    assert isinstance(sample_pos["image"], torch.Tensor)
    assert sample_pos["image"].shape[-2:] == (32, 32)
    assert len(sample_pos["heatmaps"]) == 2
    assert sample_pos["heatmaps"][0].shape == torch.Size([1, 4, 4])  # stride 8
    assert sample_pos["offsets"][0].shape == torch.Size([2, 4, 4])
    # Positive should contain a non-zero peak
    assert sample_pos["heatmaps"][0].max() > 0.5
    assert sample_pos["offsets"][0].abs().sum() > 0  # dx/dy at peak cell

    idx_neg = ds.versions.index("unknown")
    sample_neg = ds[idx_neg]
    assert sample_neg["valid_mask"].item() == 0.0
    assert all(torch.count_nonzero(h) == 0 for h in sample_neg["heatmaps"])  # empty heatmaps
    assert all(torch.count_nonzero(o) == 0 for o in sample_neg["offsets"])  # empty offsets
