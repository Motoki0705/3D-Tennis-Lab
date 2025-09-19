import numpy as np
import pytest

from development.core.data_core import targets


@pytest.mark.unit
def test_make_heatmaps_xy_marks_visible_points():
    points = [[(1.0, 1.0)]]
    heatmap = targets.make_heatmaps_xy(points, size_hw=(4, 4), sigma=1.0)
    assert heatmap.shape == (1, 1, 4, 4)
    assert np.isclose(heatmap[0, 0, 1, 1], 1.0)


@pytest.mark.unit
def test_scale_points_and_boxes():
    pts = [(2.0, 2.0), (4.0, 4.0)]
    scaled_pts = targets.scale_points(pts, source_size=(8, 8), target_size=(4, 4))
    assert scaled_pts == [(1.0, 1.0), (2.0, 2.0)]

    boxes = [(2.0, 2.0, 4.0, 4.0)]
    scaled_boxes = targets.scale_boxes(boxes, source_size=(8, 8), target_size=(4, 4))
    assert scaled_boxes == [(1.0, 1.0, 2.0, 2.0)]


@pytest.mark.unit
def test_extract_ball_keypoint_prefers_keypoints_then_bbox():
    ann_with_kp = {"keypoints": [3, 5, 2]}
    assert targets.extract_ball_keypoint(ann_with_kp) == (3.0, 5.0, 2)

    ann_with_box = {"bbox": [4, 6, 2, 2]}
    assert targets.extract_ball_keypoint(ann_with_box) == (5.0, 7.0, 2)


@pytest.mark.unit
def test_extract_player_bboxes_classes_filters_category_and_size():
    anns = [
        {"category_id": 1, "bbox": [0, 0, 5, 5]},
        {"category_id": 2, "bbox": [1, 1, 5, 5]},
        {"category_id": 1, "bbox": [2, 2, 0.5, 0.5]},
    ]
    bboxes, classes = targets.extract_player_bboxes_classes(anns, category_id=1, min_box_size=1.0)
    assert bboxes == [(0.0, 0.0, 5.0, 5.0)]
    assert classes == [1]
