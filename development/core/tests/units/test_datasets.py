import torch
import pytest

from development.core.datasets import BallSequenceDataset, PlayerSequenceDataset, CourtKeypointDataset


def _dummy_coco(images, annotations, categories):
    return {"images": images, "annotations": annotations, "categories": categories}


@pytest.mark.unit
def test_ball_sequence_dataset_builds_heatmaps(monkeypatch):
    images = [
        {"id": 1, "file_name": "clip/frame_001.png", "width": 4, "height": 4},
        {"id": 2, "file_name": "clip/frame_002.png", "width": 4, "height": 4},
    ]
    annotations = [
        {"image_id": 1, "category_id": 7, "keypoints": [1.0, 1.0, 2]},
        {"image_id": 2, "category_id": 7, "bbox": [1.0, 1.0, 1.0, 1.0]},
    ]
    categories = [{"id": 7, "name": "ball"}]
    coco = _dummy_coco(images, annotations, categories)

    monkeypatch.setattr(BallSequenceDataset, "_load_image_tensor", lambda self, frame: torch.zeros(3, 4, 4))

    dataset = BallSequenceDataset(
        coco=coco,
        sequence_length=2,
        frame_stride=1,
        heatmap_size=(4, 4),
        heatmap_sigma=1.0,
        transform=lambda sample: sample,
    )

    sample = dataset[0]
    assert sample["inputs"].shape == (2, 3, 4, 4)
    heatmaps = sample["targets"]["heatmaps"]
    assert isinstance(heatmaps, torch.Tensor)
    assert heatmaps.shape == (2, 1, 4, 4)
    assert torch.isclose(heatmaps[0, 0, 1, 1], torch.tensor(1.0))


@pytest.mark.unit
def test_player_sequence_dataset_collects_bboxes(monkeypatch):
    images = [
        {"id": 1, "file_name": "clip/frame_001.png", "width": 8, "height": 8},
        {"id": 2, "file_name": "clip/frame_002.png", "width": 8, "height": 8},
    ]
    annotations = [
        {"image_id": 1, "category_id": 5, "bbox": [0, 0, 2, 2]},
        {"image_id": 2, "category_id": 5, "bbox": [1, 1, 3, 3]},
    ]
    categories = [{"id": 5, "name": "player"}]
    coco = _dummy_coco(images, annotations, categories)

    monkeypatch.setattr(PlayerSequenceDataset, "_load_image_tensor", lambda self, frame: torch.zeros(3, 8, 8))

    dataset = PlayerSequenceDataset(
        coco=coco,
        sequence_length=2,
        frame_stride=1,
        transform=lambda sample: sample,
    )

    sample = dataset[0]
    assert sample["inputs"].shape == (2, 3, 8, 8)
    bboxes = sample["targets"]["bboxes"]
    classes = sample["targets"]["classes"]
    expected_bboxes = [[[0.0, 0.0, 2.0, 2.0]], [[1.0, 1.0, 3.0, 3.0]]]
    assert bboxes == expected_bboxes
    assert classes == [[5], [5]]


@pytest.mark.unit
def test_court_keypoint_dataset_heatmap_generation(monkeypatch):
    images = [{"id": 1, "file_name": "clip/frame_001.png", "width": 4, "height": 4}]
    annotations = [{"image_id": 1, "category_id": 1, "keypoints": [1.0, 1.0, 2, 2.0, 2.0, 2]}]
    categories = [{"id": 1, "name": "court"}]
    coco = _dummy_coco(images, annotations, categories)

    monkeypatch.setattr(CourtKeypointDataset, "_load_image_tensor", lambda self, frame: torch.zeros(3, 4, 4))

    dataset = CourtKeypointDataset(
        coco=coco,
        sequence_length=1,
        frame_stride=1,
        heatmap_size=(4, 4),
        heatmap_sigma=1.0,
        transform=lambda sample: sample,
    )

    sample = dataset[0]
    assert sample["inputs"].shape == (1, 3, 4, 4)
    heatmaps = sample["targets"]["heatmaps"]
    assert heatmaps.shape == (2, 4, 4)
    assert torch.isclose(heatmaps[0, 1, 1], torch.tensor(1.0))
