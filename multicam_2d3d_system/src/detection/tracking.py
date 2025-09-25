"""Multi-object tracking utilities."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from ..dataio import formats


@dataclass
class _Track:
    track_id: int
    cls: str
    frames: list[formats.TrackFrame] = field(default_factory=list)
    last_frame_idx: int = -1
    last_pos: tuple[float, float] | None = None
    misses: int = 0

    def add_detection(self, frame_idx: int, det: formats.Detection2DEntry) -> None:
        frame: formats.TrackFrame = {
            "frame_idx": frame_idx,
            "score": float(det.get("score", 0.0)),
        }
        if "bbox" in det and det["bbox"] is not None:
            frame["bbox"] = tuple(float(v) for v in det["bbox"])
            self.last_pos = _bbox_center(frame["bbox"])
        elif "point" in det and det["point"] is not None:
            frame["point"] = tuple(float(v) for v in det["point"])
            self.last_pos = frame["point"]  # type: ignore[assignment]
        self.frames.append(frame)
        self.last_frame_idx = frame_idx
        self.misses = 0


def track_detections(
    detections: Iterable[formats.FrameDetections2D],
    method: str = "simple_centroid",
    *,
    max_frame_gap: int = 5,
    ball_distance_px: float = 50.0,
    player_distance_px: float = 120.0,
    min_track_length: int = 3,
) -> list[formats.CameraTracks]:
    """Assign track identifiers to per-frame detections using a lightweight tracker."""

    if method not in {"simple", "simple_centroid"}:
        raise NotImplementedError(f"Tracker method '{method}' is not implemented in this demo")

    frames_by_camera: dict[str, list[formats.FrameDetections2D]] = defaultdict(list)
    for frame in detections:
        frames_by_camera[str(frame["camera_id"])].append(frame)

    results: list[formats.CameraTracks] = []
    for camera_id, frames in frames_by_camera.items():
        ordered_frames = sorted(frames, key=lambda f: int(f.get("frame_idx", 0)))
        tracks = _run_simple_tracker(
            ordered_frames,
            max_frame_gap=max_frame_gap,
            ball_distance_px=ball_distance_px,
            player_distance_px=player_distance_px,
            min_track_length=min_track_length,
        )
        results.append({
            "camera_id": camera_id,
            "tracks": tracks,
        })

    return results


def _run_simple_tracker(
    frames: list[formats.FrameDetections2D],
    *,
    max_frame_gap: int,
    ball_distance_px: float,
    player_distance_px: float,
    min_track_length: int,
) -> list[formats.TrackEntry]:
    active_tracks: dict[int, _Track] = {}
    finished_tracks: list[_Track] = []
    next_track_id = itertools.count(1)

    for frame in frames:
        frame_idx = int(frame.get("frame_idx", 0))
        detections_for_classes: dict[str, list[formats.Detection2DEntry]] = defaultdict(list)
        for det in frame.get("detections", []):
            det_cls = det.get("cls", "unknown")
            if det_cls not in {"player", "ball"}:
                continue
            if det_cls == "player" and det.get("bbox") is None:
                continue
            if det_cls == "ball" and det.get("point") is None:
                continue
            detections_for_classes[det_cls].append(det)

        # Age tracks and drop stale ones
        for track in list(active_tracks.values()):
            if frame_idx - track.last_frame_idx > max_frame_gap:
                finished_tracks.append(track)
                del active_tracks[track.track_id]

        for det_cls, cls_detections in detections_for_classes.items():
            cls_tracks = [track for track in active_tracks.values() if track.cls == det_cls]

            unmatched_tracks = {track.track_id for track in cls_tracks}
            detection_assigned: list[bool] = [False] * len(cls_detections)

            for det_idx, det in enumerate(cls_detections):
                det_pos = _detection_position(det)
                if det_pos is None:
                    continue

                best_track_id = None
                best_distance = math.inf
                for track in cls_tracks:
                    if track.track_id not in unmatched_tracks:
                        continue
                    if frame_idx - track.last_frame_idx > max_frame_gap:
                        continue
                    if track.last_pos is None:
                        continue
                    dist = _euclidean(track.last_pos, det_pos)
                    threshold = ball_distance_px if det_cls == "ball" else player_distance_px
                    if dist <= threshold and dist < best_distance:
                        best_distance = dist
                        best_track_id = track.track_id

                if best_track_id is not None:
                    track = active_tracks[best_track_id]
                    track.add_detection(frame_idx, det)
                    unmatched_tracks.discard(best_track_id)
                    detection_assigned[det_idx] = True

            # Create new tracks for unmatched detections
            for det_idx, det in enumerate(cls_detections):
                if detection_assigned[det_idx]:
                    continue
                track_id = next(next_track_id)
                new_track = _Track(track_id=track_id, cls=det_cls)
                new_track.add_detection(frame_idx, det)
                active_tracks[track_id] = new_track

        # Increment miss counters for tracks not updated in this frame
        for track in active_tracks.values():
            if track.last_frame_idx != frame_idx:
                track.misses += 1

    finished_tracks.extend(active_tracks.values())

    track_entries: list[formats.TrackEntry] = []
    for track in finished_tracks:
        if len(track.frames) < max(1, min_track_length):
            continue
        track_entries.append({
            "track_id": track.track_id,
            "cls": track.cls,
            "frames": track.frames,
        })

    return track_entries


def _detection_position(det: formats.Detection2DEntry) -> tuple[float, float] | None:
    if "point" in det and det["point"] is not None:
        point = det["point"]
        return float(point[0]), float(point[1])
    if "bbox" in det and det["bbox"] is not None:
        return _bbox_center(det["bbox"])
    return None


def _bbox_center(bbox: formats.BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = map(float, bbox)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _euclidean(pt_a: tuple[float, float], pt_b: tuple[float, float]) -> float:
    return math.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1])
