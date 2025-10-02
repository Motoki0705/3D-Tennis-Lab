"""Unified data schemas for detections, tracks, calibration, and reconstruction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

try:
    from typing import NotRequired, TypedDict
except ImportError:  # pragma: no cover - fallback for <3.11
    from typing import NotRequired  # type: ignore

    from typing_extensions import TypedDict


BBox = tuple[float, float, float, float]
Point2D = tuple[float, float]
Vector3 = tuple[float, float, float]
Matrix3x3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


class Detection2DEntry(TypedDict, total=False):
    cls: Literal["player", "ball", "court", "pose", "unknown"]
    bbox: NotRequired[BBox]
    point: NotRequired[Point2D]
    score: float
    keypoints: NotRequired[list[dict[str, float]]]


class FrameDetections2D(TypedDict):
    camera_id: str
    frame_idx: int
    timestamp: float
    detections: list[Detection2DEntry]
    image_size: tuple[int, int]


class TrackFrame(TypedDict):
    frame_idx: int
    bbox: NotRequired[BBox]
    point: NotRequired[Point2D]
    score: float


class TrackEntry(TypedDict):
    track_id: int | str
    cls: Literal["player", "ball", "unknown"]
    frames: list[TrackFrame]


class CameraTracks(TypedDict):
    camera_id: str
    tracks: list[TrackEntry]


class IntrinsicsParams(TypedDict):
    fx: float
    fy: float
    cx: float
    cy: float
    k: Sequence[float]


class ExtrinsicsParams(TypedDict):
    R: Matrix3x3
    t: Vector3


class TimestampSync(TypedDict, total=False):
    offset: float
    drift: float


class CalibrationEntry(TypedDict, total=False):
    camera_id: str
    intrinsics: IntrinsicsParams
    extrinsics: ExtrinsicsParams
    timestamp_sync: NotRequired[TimestampSync]


class Frame3DEntry(TypedDict, total=False):
    timestamp: float
    X: Vector3
    cov: NotRequired[Sequence[float]]
    src: NotRequired[dict[str, Any]]


class ObjectTrajectory(TypedDict, total=False):
    id: str | int
    frames: list[Frame3DEntry]


class CourtFrameMeta(TypedDict, total=False):
    origin: str
    axes: str


class Reconstruction3D(TypedDict, total=False):
    objects: list[ObjectTrajectory]
    court_frame: NotRequired[CourtFrameMeta]


@dataclass
class DatasetBundle:
    """Convenience wrapper for a self-consistent dataset payload."""

    detections: list[FrameDetections2D]
    tracks: list[CameraTracks]
    calibration: list[CalibrationEntry]
    reconstruction: Reconstruction3D | None = None
