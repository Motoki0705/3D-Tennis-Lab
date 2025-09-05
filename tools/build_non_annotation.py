#!/usr/bin/env python3
"""
Build unlabeled dataset frames and non_annotation.json for semi-supervised training.

What it does
- Scans source videos, splits them into fixed-length clips, extracts frames,
  and writes them under images_root as: game_{gid:03d}/Clip_{cid:04d}/frame_{idx:06d}.jpg
- Produces a COCO-like JSON (non_annotation.json) that contains only image metadata
  and minimal categories, with extra fields needed by our dataset loader:
    - original_path (relative to images_root)
    - game_id, clip_id, frame_id

Default assumptions
- Annotated games are 1..10. This tool starts unlabeled games at gid=11 by default,
  but the starting id and how many to skip are configurable.
- Source videos are located under data/raw/videos (created by tools/download).

Usage examples
  python tools/build_non_annotation.py \
      --videos-dir data/raw/videos \
      --images-root data/processed/ball/images \
      --json-out data/processed/ball/non_annotation.json \
      --start-game-id 11 --skip-first 10 --clip-len 120 --frame-stride 1

Note
- Requires OpenCV (cv2). Install if missing: pip install opencv-python
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List


def list_videos(videos_dir: Path, exts=(".mp4", ".mov", ".mkv", ".avi")) -> List[Path]:
    vids = []
    for p in sorted(videos_dir.rglob("*")):
        if p.suffix.lower() in exts:
            vids.append(p)
    return vids


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_posix_path(path: Path, relative_to: Path) -> str:
    try:
        rel = path.relative_to(relative_to)
    except Exception:
        rel = path
    return rel.as_posix()


def extract_frames_to_clips(
    video_path: Path,
    images_root: Path,
    game_id: int,
    clip_len: int = 120,
    frame_stride: int = 1,
    quality: int = 95,
) -> List[Dict]:
    """Extract frames from a single video into clip subfolders and return image entries.

    Returns a list of dicts for non_annotation.json "images" entries.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    entries: List[Dict] = []
    frame_global_idx = 0
    clip_id = 1
    frame_id_in_clip = 0

    # Create first clip dir
    clip_dir = images_root / f"game_{game_id:03d}" / f"Clip_{clip_id:04d}"
    ensure_dir(clip_dir)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if (frame_global_idx % frame_stride) != 0:
            frame_global_idx += 1
            continue

        # Roll to a new clip when clip_len is reached
        if frame_id_in_clip >= clip_len:
            clip_id += 1
            frame_id_in_clip = 0
            clip_dir = images_root / f"game_{game_id:03d}" / f"Clip_{clip_id:04d}"
            ensure_dir(clip_dir)

        # Write frame
        frame_name = f"frame_{frame_id_in_clip:06d}.jpg"
        frame_path = clip_dir / frame_name
        # High-quality JPEG
        try:
            import numpy as np  # noqa: F401 (just to ensure dependency error surfaces here if missing)

            cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        except Exception:
            # fallback without quality param
            cv2.imwrite(str(frame_path), frame)

        # Create image entry
        entry = {
            "id": None,  # will be assigned by caller
            "original_path": to_posix_path(frame_path, images_root),
            "file_name": to_posix_path(frame_path, images_root),
            "width": width,
            "height": height,
            "game_id": int(game_id),
            "clip_id": int(clip_id),
            "frame_id": int(frame_id_in_clip),
        }
        entries.append(entry)

        frame_id_in_clip += 1
        frame_global_idx += 1

    cap.release()
    return entries


def main():
    ap = argparse.ArgumentParser(description="Build unlabeled dataset and non_annotation.json from videos.")
    ap.add_argument("--videos-dir", type=str, default="data/raw/videos", help="Directory containing source videos")
    ap.add_argument(
        "--images-root", type=str, default="data/processed/ball/images", help="Root dir to write extracted frames"
    )
    ap.add_argument(
        "--json-out", type=str, default="data/processed/ball/non_annotation.json", help="Path to output JSON"
    )
    ap.add_argument("--start-game-id", type=int, default=11, help="Starting game id for unlabeled videos")
    ap.add_argument("--skip-first", type=int, default=10, help="Skip first N videos (assumed labeled)")
    ap.add_argument("--clip-len", type=int, default=120, help="Frames per clip directory")
    ap.add_argument("--frame-stride", type=int, default=1, help="Sample every N-th frame")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of videos processed (0=all)")
    args = ap.parse_args()

    videos_dir = Path(args.videos_dir)
    images_root = Path(args.images_root)
    json_out = Path(args.json_out)
    ensure_dir(images_root)
    ensure_dir(json_out.parent)

    videos = list_videos(videos_dir)
    if not videos:
        print(f"No videos found under: {videos_dir}")
        return

    # Skip first N (annotated) and process the rest
    vids = videos[args.skip_first :]
    if args.limit > 0:
        vids = vids[: args.limit]

    images: List[Dict] = []
    categories = [{"id": 1, "name": "ball", "keypoints": ["center"], "skeleton": []}]

    next_image_id = 1
    game_id = int(args.start_game_id)
    for vi, vpath in enumerate(vids):
        print(f"[ {vi + 1}/{len(vids)} ] Processing video: {vpath}")
        try:
            entries = extract_frames_to_clips(
                vpath,
                images_root,
                game_id=game_id,
                clip_len=int(args.clip_len),
                frame_stride=int(args.frame_stride),
            )
        except Exception as e:
            print(f"  - Failed: {e}")
            continue

        # Assign global image ids
        for e in entries:
            e["id"] = next_image_id
            next_image_id += 1
        images.extend(entries)

        game_id += 1

    out_data = {
        "images": images,
        "annotations": [],  # unlabeled
        "categories": categories,
    }

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False)

    print(f"Wrote {len(images)} image entries to {json_out}")


if __name__ == "__main__":
    main()
