"""Target construction utilities shared by core datasets."""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]


def make_heatmaps_xy(
    points_per_frame: Sequence[Sequence[Optional[Point]]],
    *,
    size_hw: Tuple[int, int],
    sigma: float,
    visibility: Optional[Sequence[Sequence[bool]]] = None,
    dtype=np.float32,
) -> np.ndarray:
    """Generate Gaussian heatmaps for per-frame keypoints.

    Parameters
    ----------
    points_per_frame:
        Sequence of length ``T`` where each entry is a sequence of ``K`` points
        (``None`` to mark missing points).
    size_hw:
        Output heatmap size ``(H, W)``.
    sigma:
        Standard deviation of the Gaussian (in heatmap pixels).
    visibility:
        Optional boolean mask with the same ``T × K`` structure indicating
        whether each keypoint should contribute to the heatmap.
    dtype:
        ``numpy`` dtype of the returned tensor (defaults to ``np.float32``).
    """

    T = len(points_per_frame)
    K = max((len(frame) for frame in points_per_frame), default=0)
    if K == 0 or T == 0:
        return np.zeros((T, 0, *size_hw), dtype=dtype)

    heatmaps = np.zeros((T, K, size_hw[0], size_hw[1]), dtype=dtype)
    xs = np.arange(size_hw[1], dtype=np.float32)[None, :]
    ys = np.arange(size_hw[0], dtype=np.float32)[:, None]
    sigma_sq = np.float32(2 * sigma * sigma)

    for t, frame_points in enumerate(points_per_frame):
        frame_vis = visibility[t] if visibility is not None and t < len(visibility) else None
        for k in range(K):
            point = frame_points[k] if k < len(frame_points) else None
            if point is None:
                continue
            if frame_vis is not None and (k >= len(frame_vis) or not frame_vis[k]):
                continue
            x, y = point
            g = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / sigma_sq)
            heatmaps[t, k] = g
    return heatmaps


def scale_points(
    points: Sequence[Optional[Point]] | np.ndarray,
    *,
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> List[Optional[Point]]:
    """Scale points from ``source_size`` (H,W) to ``target_size`` (H,W)."""

    if len(points) == 0:
        return []  # type: ignore[return-value]
    h_src, w_src = map(float, source_size)
    h_tgt, w_tgt = map(float, target_size)
    scale_x = w_tgt / max(w_src, 1.0)
    scale_y = h_tgt / max(h_src, 1.0)

    def _convert(pt: Optional[Point]) -> Optional[Point]:
        if pt is None:
            return None
        x, y = pt
        return (float(x) * scale_x, float(y) * scale_y)

    return [_convert(pt) for pt in points]


def scale_boxes(
    boxes: Sequence[BBox] | np.ndarray,
    *,
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> List[BBox]:
    """Scale COCO-format boxes from ``source_size`` to ``target_size``."""

    h_src, w_src = map(float, source_size)
    h_tgt, w_tgt = map(float, target_size)
    scale_x = w_tgt / max(w_src, 1.0)
    scale_y = h_tgt / max(h_src, 1.0)

    scaled: List[BBox] = []
    for box in boxes:
        x, y, w, h = map(float, box)
        scaled.append((x * scale_x, y * scale_y, w * scale_x, h * scale_y))
    return scaled


def extract_ball_keypoint(ann: Mapping[str, Any]) -> Optional[tuple[float, float, int]]:
    """Extract a single ball keypoint (x, y, visibility) from a COCO annotation."""

    keypoints = ann.get("keypoints")
    if isinstance(keypoints, Sequence) and len(keypoints) >= 3:
        try:
            x, y, v = keypoints[:3]
            return float(x), float(y), int(v)
        except (TypeError, ValueError):
            return None
    bbox = ann.get("bbox")
    if isinstance(bbox, Sequence) and len(bbox) == 4:
        try:
            x, y, w, h = bbox
            return float(x) + float(w) / 2.0, float(y) + float(h) / 2.0, 2
        except (TypeError, ValueError):
            return None
    return None


def extract_player_bboxes_classes(
    annotations: Iterable[Mapping[str, Any]],
    *,
    category_id: int,
    min_box_size: float = 1.0,
) -> tuple[List[BBox], List[int]]:
    """Return COCO-format boxes and class labels filtered by ``category_id``."""

    bboxes: List[BBox] = []
    classes: List[int] = []
    for ann in annotations:
        if int(ann.get("category_id", -1)) != int(category_id):
            continue
        bbox = ann.get("bbox")
        if not isinstance(bbox, Sequence) or len(bbox) != 4:
            continue
        x, y, w, h = bbox
        if float(w) < min_box_size or float(h) < min_box_size:
            continue
        bboxes.append((float(x), float(y), float(w), float(h)))
        classes.append(int(ann.get("category_id", category_id)))
    return bboxes, classes


def infer_sequence_keypoints(
    frames: Sequence[Mapping[str, Any]],
    *,
    keypoint_key: str = "keypoints",
    visibility_key: str = "visibility",
) -> tuple[List[List[Optional[Point]]], List[List[bool]]]:
    """Collect per-frame keypoints/visibility from enriched frame records."""

    points: List[List[Optional[Point]]] = []
    visibility: List[List[bool]] = []
    for frame in frames:
        coords = frame.get(keypoint_key)
        vis = frame.get(visibility_key)
        if isinstance(coords, Sequence) and isinstance(vis, Sequence):
            frame_pts = []
            frame_vis = []
            for xy, v in zip(coords, vis):
                if xy is None:
                    frame_pts.append(None)
                    frame_vis.append(bool(v))
                    continue
                if isinstance(xy, Sequence) and len(xy) >= 2:
                    frame_pts.append((float(xy[0]), float(xy[1])))
                else:
                    frame_pts.append(None)
                frame_vis.append(bool(v))
        elif isinstance(coords, Sequence) and len(coords) >= 2:
            frame_pts = [(float(coords[0]), float(coords[1]))]
            frame_vis = [bool(vis) if vis is not None else True]
        else:
            frame_pts = []
            frame_vis = []
        points.append(frame_pts)
        visibility.append(frame_vis)
    return points, visibility


__all__ = [
    "BBox",
    "Point",
    "extract_ball_keypoint",
    "extract_player_bboxes_classes",
    "infer_sequence_keypoints",
    "make_heatmaps_xy",
    "scale_boxes",
    "scale_points",
]
