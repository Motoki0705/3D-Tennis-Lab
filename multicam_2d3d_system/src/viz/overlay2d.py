"""2D overlay visualization helpers."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..dataio import formats

_LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import cv2
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    cv2 = None  # type: ignore[assignment]


_DEFAULT_COLORS = {
    "player": (0, 196, 255),
    "ball": (0, 255, 0),
    "court": (255, 128, 0),
    "pose": (255, 0, 191),
    "unknown": (200, 200, 200),
}


def _get_color(label: str) -> tuple[int, int, int]:
    return _DEFAULT_COLORS.get(label, (255, 255, 255))


def _resolve_canvas_size(
    payload: formats.FrameDetections2D,
    capture: Any | None,
) -> tuple[int, int]:
    image_size = payload.get("image_size")
    if isinstance(image_size, list | tuple) and len(image_size) == 2:
        width = int(image_size[0])
        height = int(image_size[1])
        if width > 0 and height > 0:
            return width, height
    if capture is not None and cv2 is not None:  # pragma: no branch - depends on optional import
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        return width, height
    return 1920, 1080


def _prepare_blank(width: int, height: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _draw_detections(frame_img: np.ndarray, detections: Iterable[formats.Detection2DEntry]) -> None:
    if cv2 is None:  # pragma: no cover - safeguard when OpenCV missing
        return
    for det in detections:
        label = det.get("cls", "unknown")
        color = _get_color(label)
        bbox = det.get("bbox")
        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame_img, (x1, y1), (x2, y2), color, 2)
        point = det.get("point")
        if point is not None and len(point) == 2:
            cx, cy = map(int, point)
            cv2.circle(frame_img, (cx, cy), 4, color, -1)
        for kp in det.get("keypoints", []) or []:
            x = int(kp.get("x", kp.get("u", 0)))
            y = int(kp.get("y", kp.get("v", 0)))
            score = float(kp.get("score", 1.0))
            if score > 0:
                cv2.circle(frame_img, (x, y), 2, color, -1)
        caption = label
        score = det.get("score")
        if isinstance(score, float | int):
            caption = f"{caption}:{score:.2f}"
        if bbox is not None and len(bbox) == 4:
            x1, y1, _, _ = map(int, bbox)
            cv2.putText(
                frame_img,
                caption,
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )


def _index_tracks(
    tracks: Iterable[formats.CameraTracks] | None,
) -> dict[str, dict[int, list[tuple[formats.TrackEntry, formats.TrackFrame]]]]:
    frame_index: dict[str, dict[int, list[tuple[formats.TrackEntry, formats.TrackFrame]]]] = {}
    if not tracks:
        return frame_index
    for bundle in tracks:
        camera_id = str(bundle.get("camera_id", ""))
        if not camera_id:
            continue
        per_frame: dict[int, list[tuple[formats.TrackEntry, formats.TrackFrame]]] = defaultdict(list)
        for track in bundle.get("tracks", []):
            for frame_entry in track.get("frames", []):
                frame_idx = int(frame_entry.get("frame_idx", -1))
                if frame_idx < 0:
                    continue
                per_frame[frame_idx].append((track, frame_entry))
        frame_index[camera_id] = per_frame
    return frame_index


def _draw_tracks(
    frame_img: np.ndarray,
    track_entries: list[tuple[formats.TrackEntry, formats.TrackFrame]],
) -> None:
    if cv2 is None:  # pragma: no cover - safeguard when OpenCV missing
        return
    for track, frame_entry in track_entries:
        label = str(track.get("track_id", "track"))
        cls = track.get("cls", "unknown")
        color = _get_color(cls)
        bbox = frame_entry.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame_img, (x1, y1), (x2, y2), color, 1)
        point = frame_entry.get("point")
        if point and len(point) == 2:
            cx, cy = map(int, point)
            cv2.circle(frame_img, (cx, cy), 3, color, -1)
        bbox = frame_entry.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, _, _ = map(int, bbox)
            cv2.putText(
                frame_img,
                label,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )


def render_overlays(
    frames: Iterable[formats.FrameDetections2D],
    output_dir: Path,
    *,
    video_sources: Mapping[str, Path] | None = None,
    tracks: Iterable[formats.CameraTracks] | None = None,
    codec: str = "mp4v",
    fps_fallback: float = 30.0,
) -> None:
    """Render bounding boxes and tracks onto per-camera overlay videos."""

    if cv2 is None:  # pragma: no cover - avoid hard failure when OpenCV missing
        _LOGGER.warning("OpenCV is not installed; skipping overlay rendering")
        return

    grouped: dict[str, list[formats.FrameDetections2D]] = defaultdict(list)
    for payload in frames:
        camera_id = str(payload.get("camera_id", "unknown"))
        grouped[camera_id].append(payload)

    if not grouped:
        _LOGGER.info("No frames provided for overlay rendering")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    track_index = _index_tracks(tracks)

    for camera_id, camera_frames in grouped.items():
        camera_frames.sort(key=lambda frame: int(frame.get("frame_idx", 0)))
        source_path = None
        capture = None
        if video_sources:
            source_path = Path(video_sources.get(camera_id, ""))
            if source_path and source_path.exists():
                capture = cv2.VideoCapture(str(source_path))
            else:
                source_path = None

        fps = fps_fallback
        writer = None
        width = height = 0

        try:
            for payload in camera_frames:
                frame_idx = int(payload.get("frame_idx", 0))
                if capture is not None:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ok, frame_img = capture.read()
                    if not ok or frame_img is None:
                        frame_img = None
                else:
                    frame_img = None

                width, height = _resolve_canvas_size(payload, capture)
                if frame_img is None:
                    frame_img = _prepare_blank(width, height)

                if capture is not None:
                    frame_fps = capture.get(cv2.CAP_PROP_FPS)
                    if frame_fps and frame_fps > 0:
                        fps = frame_fps

                _draw_detections(frame_img, payload.get("detections", []))

                track_entries = track_index.get(camera_id, {}).get(frame_idx, [])
                if track_entries:
                    _draw_tracks(frame_img, track_entries)

                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    output_path = output_dir / f"{camera_id}_overlay.mp4"
                    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                    if not writer.isOpened():  # pragma: no cover - guard against backend issues
                        raise RuntimeError(f"Failed to open video writer for {output_path}")

                writer.write(frame_img)
        finally:
            if capture is not None:
                capture.release()
            if writer is not None:
                writer.release()

        if writer is None:
            _LOGGER.info("Camera %s had no frames to render", camera_id)
        else:
            _LOGGER.info("Rendered overlay video for camera %s to %s", camera_id, output_dir)
