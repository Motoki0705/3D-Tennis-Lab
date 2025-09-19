"""Grouping helpers shared by sequential datasets (ball/player/etc.)."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence

FrameRecord = Mapping[str, Any]
ClipRecord = Dict[str, Any]


def group_clips(
    frames: Iterable[FrameRecord],
    *,
    game_key: str = "game_id",
    clip_key: str = "clip_id",
    path_key: str = "file_name",
    original_path_key: str = "original_path",
) -> List[ClipRecord]:
    """Cluster frames into clips using metadata or directory heuristics.

    Parameters
    ----------
    frames:
        Iterable of image metadata dictionaries (typically COCO ``images`` entries
        enriched with dataset-specific fields).
    game_key, clip_key:
        When both keys are present on **all** frames, they are used to create
        deterministic clip ids.
    path_key, original_path_key:
        Fallback file path keys used to group frames by parent directory when the
        explicit keys are missing.
    """

    records = list(frames)
    if not records:
        return []

    have_ids = all(game_key in frame and clip_key in frame for frame in records)
    clips: List[ClipRecord] = []

    if have_ids:
        grouped: Dict[tuple[int, int], List[FrameRecord]] = {}
        for frame in records:
            key = (int(frame[game_key]), int(frame[clip_key]))
            grouped.setdefault(key, []).append(frame)
        for (game_id, clip_id), frames_in_clip in grouped.items():
            clips.append({
                "game_id": game_id,
                "clip_id": clip_id,
                "frames": _sort_frames(frames_in_clip, path_key, original_path_key),
            })
        clips.sort(key=lambda rec: (rec["game_id"], rec["clip_id"]))
        return clips

    grouped: Dict[str, List[FrameRecord]] = {}
    for frame in records:
        path = _safe_path(frame, path_key=path_key, original_path_key=original_path_key)
        parent = os.path.dirname(path)
        grouped.setdefault(parent, []).append(frame)

    for idx, (parent, frames_in_clip) in enumerate(sorted(grouped.items())):
        clips.append({
            "group": parent,
            "clip_index": idx,
            "frames": _sort_frames(frames_in_clip, path_key, original_path_key),
        })
    return clips


def enumerate_sequences(
    total_frames: int,
    *,
    sequence_length: int,
    frame_stride: int,
    allow_partial_last: bool = False,
    drop_short_clips: bool = False,
) -> List[List[int]]:
    """Enumerate sliding-window frame indices for sequential sampling."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive.")
    if total_frames <= 0:
        return []

    indices: List[List[int]] = []
    stop = total_frames - sequence_length + 1
    if stop > 0:
        for start in range(0, stop, frame_stride):
            seq = list(range(start, start + sequence_length))
            indices.append(seq)

    remainder = total_frames % frame_stride
    last_start = total_frames - sequence_length
    if last_start < 0:
        last_start = 0

    if allow_partial_last and (not indices or indices[-1][-1] < total_frames - 1):
        tail = list(range(total_frames - min(sequence_length, total_frames), total_frames))
        if len(tail) == sequence_length:
            indices.append(tail)
        elif not drop_short_clips:
            indices.append(tail)

    if not indices and not drop_short_clips:
        indices.append(list(range(total_frames)))

    if drop_short_clips:
        indices = [seq for seq in indices if len(seq) == sequence_length]

    return indices


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sort_frames(
    frames: Sequence[FrameRecord],
    path_key: str,
    original_path_key: str,
) -> List[FrameRecord]:
    return sorted(
        frames,
        key=lambda rec: (
            _natural_key(_safe_path(rec, path_key=path_key, original_path_key=original_path_key)),
            _coerce_frame_index(rec.get("frame_id")),
        ),
    )


def _natural_key(path: str) -> List[Any]:
    key: List[Any] = []
    for chunk in re.findall(r"\d+|\D+", path):
        if chunk.isdigit():
            key.append((0, int(chunk)))
        else:
            key.append((1, chunk))
    return key


def _safe_path(frame: Mapping[str, Any], *, path_key: str, original_path_key: str) -> str:
    return str(frame.get(original_path_key) or frame.get(path_key) or "")


def _coerce_frame_index(value: Any) -> int:
    if value is None:
        return int(1e12)
    try:
        return int(value)
    except (TypeError, ValueError):
        if isinstance(value, str):
            match = re.search(r"\d+", value)
            if match:
                try:
                    return int(match.group(0))
                except ValueError:
                    return int(1e12)
        return int(1e12)


__all__ = ["ClipRecord", "FrameRecord", "enumerate_sequences", "group_clips"]
