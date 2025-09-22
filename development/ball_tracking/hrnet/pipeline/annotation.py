from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
from .clip_extractor.base import Clip, ClipFrame

LOG = logging.getLogger(__name__)

CLIP_DIR_PATTERN = re.compile(r"Clip(\d+)")


@dataclass
class ExportPaths:
    images_root: Path
    ann_clips_root: Path


class VideoFrameExtractor:
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video {video_path}")
        self.current_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self, frame_idx: int):
        if frame_idx != self.current_index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self.current_index = frame_idx
        ret, frame = self.cap.read()
        if not ret:
            raise IOError(f"Failed to read frame {frame_idx} from {self.video_path}")
        self.current_index += 1
        return frame

    def release(self) -> None:
        self.cap.release()


def determine_next_clip_index(game_images_dir: Path) -> int:
    if not game_images_dir.exists():
        return 1
    indices = [
        int(match.group(1)) for path in game_images_dir.glob("Clip*") if (match := CLIP_DIR_PATTERN.match(path.name))
    ]
    return max(indices, default=0) + 1


def export_clip(
    clip: Clip,
    *,
    clip_index: int,
    game_id: str,
    video_id: str,
    reader: VideoFrameExtractor,
    export_paths: ExportPaths,
    jpg_quality: int,
    write_frames: bool = True,
) -> Path:
    clip_name = f"Clip{clip_index}"
    image_dir = export_paths.images_root / game_id / clip_name
    image_dir.mkdir(parents=True, exist_ok=True)

    ann_path = export_paths.ann_clips_root / game_id
    ann_path.mkdir(parents=True, exist_ok=True)
    ann_file = ann_path / f"{clip_name}.json"

    frame_lookup: Dict[int, ClipFrame] = {frame.frame_idx: frame for frame in clip.frames}

    start = clip.start_frame
    end = clip.end_frame
    annotations: List[dict] = []
    images_meta: List[dict] = []

    frame_counter = 0

    for frame_idx in range(start, end + 1):
        frame = reader.read(frame_idx) if write_frames else None
        file_name = f"{frame_counter:04d}.jpg"
        output_path = image_dir / file_name
        if write_frames:
            cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])

        clip_frame = frame_lookup.get(frame_idx)
        has_ball = clip_frame is not None
        visibility = 2 if has_ball else 0
        x = float(clip_frame.xc) if has_ball else 0.0
        y = float(clip_frame.yc) if has_ball else 0.0
        conf = float(clip_frame.conf) if has_ball else 0.0

        images_meta.append({
            "id": frame_counter + 1,
            "file_name": file_name,
            "original_path": f"{game_id}/{clip_name}/{file_name}",
            "width": reader.width,
            "height": reader.height,
            "license": 1,
            "game_id": game_id,
            "clip_id": clip_index,
        })

        keypoints = [x, y, visibility]
        annotations.append({
            "id": (clip_index * 10000) + frame_counter + 1,
            "image_id": frame_counter + 1,
            "category_id": 1,
            "keypoints": keypoints,
            "num_keypoints": 1 if has_ball else 0,
            "attributes": {"conf": conf},
        })

        frame_counter += 1

    clip_json = {
        "info": {"description": "tennis-ball clip"},
        "categories": [
            {
                "id": 1,
                "name": "tennis_ball",
                "keypoints": ["center"],
                "skeleton": [],
            }
        ],
        "images": images_meta,
        "annotations": annotations,
        "metadata": {
            "video_id": video_id,
            "start_frame": start,
            "end_frame": end,
        },
        "accepted": False,
    }

    with ann_file.open("w", encoding="utf-8") as f:
        json.dump(clip_json, f, ensure_ascii=False, indent=2)

    LOG.info("Wrote clip %s with %d frames", ann_file, frame_counter)
    return ann_file


def load_clip_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_clip_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def iter_clip_json(ann_clips_root: Path) -> Iterable[Path]:
    if not ann_clips_root.exists():
        return []
    return sorted(ann_clips_root.rglob("Clip*.json"))


def merge_accepted_clips(ann_paths: Iterable[Path], output_path: Path) -> None:
    images: List[dict] = []
    annotations: List[dict] = []
    next_image_id = 1
    next_ann_id = 1

    for path in ann_paths:
        data = load_clip_json(path)
        if not data.get("accepted", False):
            continue
        id_map: Dict[int, int] = {}
        for image in data.get("images", []):
            new_id = next_image_id
            id_map[int(image["id"])] = new_id
            images.append({**image, "id": new_id})
            next_image_id += 1

        for anno in data.get("annotations", []):
            original_image_id = int(anno["image_id"])
            if original_image_id not in id_map:
                continue
            annotations.append({
                **anno,
                "id": next_ann_id,
                "image_id": id_map[original_image_id],
            })
            next_ann_id += 1

    if not images:
        LOG.info("No accepted clips found; skipping final merge")
        return

    merged = {
        "info": {"description": "accepted tennis-ball clips"},
        "categories": [
            {
                "id": 1,
                "name": "tennis_ball",
                "keypoints": ["center"],
                "skeleton": [],
            }
        ],
        "images": images,
        "annotations": annotations,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    LOG.info("Merged %d images into %s", len(images), output_path)
