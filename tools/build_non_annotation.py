#!/usr/bin/env python3
"""
Build unlabeled dataset frames and non_annotation.json for semi-supervised training.

Modes
- Batch (existing): fixed-length splitting across entire videos.
- Interactive (new): manual UI to set clip start_i / end_i, extract frames
  per selection into Clip_{i}/, and append metadata to non_annotation.json.

Key goals
- 1 video -> 1 game_{i}/ mapping with idempotent resume.
- Never re-split the same video twice; persist progress in a state file.
- "Next video" without stamping to the last frame, and resume after interrupt.

Default assumptions
- Annotated games are 1..10. This tool starts unlabeled games at gid=11 by default,
  but the starting id is configurable.
- Source videos are located under data/raw/videos (created by tools/download).

Batch usage (unchanged)
  python tools/build_non_annotation.py \
      --videos-dir data/raw/videos \
      --images-root data/processed/ball/images \
      --json-out data/processed/ball/non_annotation.json \
      --start-game-id 11 --skip-first 10 --clip-len 120 --frame-stride 1

Interactive usage
  python tools/build_non_annotation.py --interactive \
      --videos-dir data/raw/videos \
      --images-root data/processed/ball/images \
      --json-out data/processed/ball/non_annotation.json \
      --state-file data/processed/ball/non_annotation_state.json \
      --start-game-id 11 --frame-stride 1

Interactive UI keys
  Space: play/pause  |  a/d: -1/+1 frame  |  j/l: -25/+25 frames  |  J/L: -250/+250
  s: mark start      |  e: mark end and extract clip (start..cur)
  n: next video (mark done)  |  r: reset start/end  |  q: save+quit

Notes
- Requires OpenCV (cv2). Install: pip install opencv-python
- Downloader duplication guard: tools/download/youtube_downloader.py uses archive
  to avoid duplicate downloads. This tool ties each unique source video to a unique
  game_{i} directory via the persistent state mapping.
"""

from __future__ import annotations
import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


# ------------------------------
# Persistent state (idempotency & resume)
# ------------------------------


@dataclass
class VideoState:
    video_path: str  # stored as posix relative to videos_root if possible
    video_key: str  # stable key (e.g., YouTube ID or filename)
    game_id: int
    next_clip_id: int
    status: str  # 'in_progress' | 'done'
    last_frame: int = 0
    processed_ranges: List[Tuple[int, int]] = None  # inclusive start..end


def extract_youtube_id(name: str) -> Optional[str]:
    """Extract YouTube-like 11-char ID from filename if present.
    Matches patterns like '...-dQw4w9WgXcQ.mp4'. Returns None if not found.
    """
    m = re.search(r"-([A-Za-z0-9_-]{11})(?:\.|$)", name)
    return m.group(1) if m else None


def make_video_key(p: Path) -> str:
    vid = extract_youtube_id(p.name)
    if vid:
        return f"yt:{vid}"
    # fallback: filename stem (including parent dir name to reduce collisions)
    try:
        parent = p.parent.name
        return f"fs:{parent}/{p.stem}"
    except Exception:
        return f"fs:{p.stem}"


def rel_to(root: Path, p: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except Exception:
        return p.as_posix()


def load_state(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        return {
            "next_game_id": None,  # assigned on first use
            "next_image_id": None,  # assigned on first JSON init/read
            "video_map": {},  # video_key -> VideoState as dict
            "images_root": None,
            "videos_root": None,
            "json_out": None,
        }
    with state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state_path: Path, state: Dict[str, Any]) -> None:
    ensure_dir(state_path.parent)
    tmp = state_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp.replace(state_path)


def ensure_json_initialized(json_out: Path) -> Dict[str, Any]:
    """Create empty unlabeled JSON if missing; return parsed JSON dict.
    Structure: { images: [], annotations: [], categories: [...] }
    """
    if json_out.exists():
        with json_out.open("r", encoding="utf-8") as f:
            return json.load(f) or {}
    ensure_dir(json_out.parent)
    data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "ball", "keypoints": ["center"], "skeleton": []}],
    }
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return data


def append_images_to_json(json_out: Path, images_to_add: List[Dict[str, Any]]) -> int:
    data = ensure_json_initialized(json_out)
    images = data.get("images") or []
    images.extend(images_to_add)
    data["images"] = images
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return len(images_to_add)


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


def extract_frames_range(
    video_path: Path,
    images_root: Path,
    game_id: int,
    clip_id: int,
    start_frame: int,
    end_frame: int,
    frame_stride: int = 1,
    quality: int = 95,
) -> List[Dict[str, Any]]:
    """Extract frames from start_frame..end_frame (inclusive) into a specific clip directory.

    Returns a list of image entries (id unassigned) to append to non_annotation.json.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, int(start_frame))
    end = min(total - 1, int(end_frame))
    if end < start:
        cap.release()
        return []

    clip_dir = images_root / f"game_{game_id:03d}" / f"Clip_{clip_id:04d}"
    ensure_dir(clip_dir)

    # Random access to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    fidx = start
    written = 0
    entries: List[Dict[str, Any]] = []
    while fidx <= end:
        ok, frame = cap.read()
        if not ok:
            break
        # respect stride
        if ((fidx - start) % frame_stride) != 0:
            fidx += 1
            continue
        frame_id_in_clip = (fidx - start) // max(1, frame_stride)
        frame_name = f"frame_{frame_id_in_clip:06d}.jpg"
        frame_path = clip_dir / frame_name
        try:
            import numpy as np  # noqa: F401

            cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        except Exception:
            cv2.imwrite(str(frame_path), frame)

        entries.append({
            "id": None,
            "original_path": to_posix_path(frame_path, images_root),
            "file_name": to_posix_path(frame_path, images_root),
            "width": width,
            "height": height,
            "game_id": int(game_id),
            "clip_id": int(clip_id),
            "frame_id": int(frame_id_in_clip),
        })

        written += 1
        fidx += 1

    cap.release()
    return entries


def interactive_loop(
    videos: List[Path],
    videos_root: Path,
    images_root: Path,
    json_out: Path,
    state_path: Path,
    start_game_id: int,
    frame_stride: int,
    quality: int = 95,
) -> None:
    import cv2

    state = load_state(state_path)
    state["images_root"] = str(images_root.as_posix())
    state["videos_root"] = str(videos_root.as_posix())
    state["json_out"] = str(json_out.as_posix())

    # Ensure JSON exists and determine next_image_id
    data = ensure_json_initialized(json_out)
    if state.get("next_image_id") is None:
        ids = [int(im.get("id", 0)) for im in data.get("images", [])]
        state["next_image_id"] = (max(ids) + 1) if ids else 1

    # Initialize next_game_id if missing: start_game_id or max existing + 1
    if state.get("next_game_id") is None:
        # Try to infer from existing directories under images_root
        existing = []
        if images_root.exists():
            for d in images_root.glob("game_*"):
                try:
                    existing.append(int(str(d.name).split("_")[-1]))
                except Exception:
                    pass
        state["next_game_id"] = max(existing) + 1 if existing else int(start_game_id)

    # Prep window
    win = "non-annotation clipper"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def goto_frame(cap: Any, idx: int) -> Tuple[bool, Optional[Any]]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        return ok, frame

    try:
        for vpath in videos:
            vkey = make_video_key(vpath)
            vrel = rel_to(videos_root, vpath)

            # Skip if done
            vm = state.get("video_map", {})
            vst: Dict[str, Any] = vm.get(vkey)
            if vst and vst.get("status") == "done":
                continue

            # Assign game_id if new
            if not vst:
                gid = int(state["next_game_id"])
                vst = {
                    "video_path": vrel,
                    "video_key": vkey,
                    "game_id": gid,
                    "next_clip_id": 1,
                    "status": "in_progress",
                    "last_frame": 0,
                    "processed_ranges": [],
                }
                state.setdefault("video_map", {})[vkey] = vst
                state["next_game_id"] = gid + 1
                save_state(state_path, state)

            # Open video
            cap = cv2.VideoCapture(str(vpath))
            if not cap.isOpened():
                print(f"Failed to open: {vpath}")
                continue

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            cur = max(0, int(vst.get("last_frame", 0)))
            start_sel: Optional[int] = None
            playing = False

            # Trackbar for quick seek
            def on_seek(val: int):
                nonlocal cur
                cur = int(val)

            cv2.createTrackbar("pos", win, min(cur, max(0, total - 1)), max(0, total - 1), on_seek)

            # initial frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
            ok, frame = cap.read()
            if not ok:
                cur = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
                ok, frame = cap.read()
            if not ok:
                cap.release()
                print(f"Cannot read from: {vpath}")
                continue

            while True:
                # overlay HUD
                hud = f"game={vst['game_id']} clip=#{vst['next_clip_id']:04d} frame={cur}/{total - 1}"
                if start_sel is not None:
                    hud += f" | start={start_sel}"
                hud2 = "[Space]Play [a/d]±1 [j/l]±25 [J/L]±250 | s=start e=end->save | n=next r=reset q=quit"
                vis = frame.copy()
                try:
                    import cv2 as _cv2

                    _cv2.putText(vis, hud, (10, 24), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    _cv2.putText(vis, hud2, (10, 48), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                except Exception:
                    pass
                cv2.imshow(win, vis)

                key = cv2.waitKey(int(1 if playing else 30)) & 0xFF
                if playing:
                    cur = min(total - 1, cur + 1)
                    ok, frame = goto_frame(cap, cur)
                    if not ok:
                        playing = False
                        cur = max(0, min(cur, total - 1))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
                        ok, frame = cap.read()
                    cv2.setTrackbarPos("pos", win, cur)
                    continue

                if key == ord(" "):
                    playing = not playing
                elif key in (ord("a"), 81):  # left
                    cur = max(0, cur - 1)
                    ok, frame = goto_frame(cap, cur)
                    cv2.setTrackbarPos("pos", win, cur)
                elif key in (ord("d"), 83):  # right
                    cur = min(total - 1, cur + 1)
                    ok, frame = goto_frame(cap, cur)
                    cv2.setTrackbarPos("pos", win, cur)
                elif key == ord("j"):
                    cur = max(0, cur - 25)
                    ok, frame = goto_frame(cap, cur)
                    cv2.setTrackbarPos("pos", win, cur)
                elif key == ord("l"):
                    cur = min(total - 1, cur + 25)
                    ok, frame = goto_frame(cap, cur)
                    cv2.setTrackbarPos("pos", win, cur)
                elif key == ord("J"):
                    cur = max(0, cur - 250)
                    ok, frame = goto_frame(cap, cur)
                    cv2.setTrackbarPos("pos", win, cur)
                elif key == ord("L"):
                    cur = min(total - 1, cur + 250)
                    ok, frame = goto_frame(cap, cur)
                    cv2.setTrackbarPos("pos", win, cur)
                elif key == ord("s"):
                    start_sel = cur
                elif key == ord("r"):
                    start_sel = None
                elif key == ord("e"):
                    if start_sel is None:
                        continue
                    s_f = int(min(start_sel, cur))
                    e_f = int(max(start_sel, cur))
                    # Avoid duplicate ranges
                    if any((s_f <= b and a <= e_f) for (a, b) in vst.get("processed_ranges", [])):
                        print("Range overlaps processed range. Skipped.")
                    else:
                        # Extract & append JSON
                        entries = extract_frames_range(
                            vpath,
                            images_root,
                            game_id=int(vst["game_id"]),
                            clip_id=int(vst["next_clip_id"]),
                            start_frame=s_f,
                            end_frame=e_f,
                            frame_stride=int(frame_stride),
                            quality=int(quality),
                        )
                        # Assign IDs
                        nid = int(state["next_image_id"])
                        for e in entries:
                            e["id"] = nid
                            nid += 1
                        if entries:
                            appended = append_images_to_json(json_out, entries)
                            state["next_image_id"] = nid
                            vst.setdefault("processed_ranges", []).append([s_f, e_f])
                            vst["next_clip_id"] = int(vst["next_clip_id"]) + 1
                            save_state(state_path, state)
                            print(f"Saved clip: frames {s_f}..{e_f} ({appended} images)")
                    start_sel = None
                elif key == ord("n"):
                    # Mark done and go next
                    vst["status"] = "done"
                    vst["last_frame"] = cur
                    state["video_map"][vkey] = vst
                    save_state(state_path, state)
                    break
                elif key == ord("q"):
                    vst["last_frame"] = cur
                    state["video_map"][vkey] = vst
                    save_state(state_path, state)
                    cap.release()
                    cv2.destroyWindow(win)
                    return

                # update trackbar and persist last frame periodically
                cv2.setTrackbarPos("pos", win, cur)
                vst["last_frame"] = cur

            cap.release()
    finally:
        try:
            import cv2 as _cv2

            _cv2.destroyWindow(win)
        except Exception:
            pass


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
    ap.add_argument("--skip-first", type=int, default=10, help="Skip first N videos (assumed labeled) [batch mode]")
    ap.add_argument("--clip-len", type=int, default=120, help="Frames per clip directory [batch mode]")
    ap.add_argument("--frame-stride", type=int, default=1, help="Sample every N-th frame")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of videos processed (0=all) [batch mode]")
    ap.add_argument("--interactive", action="store_true", help="Use interactive UI to select clips manually")
    ap.add_argument(
        "--state-file",
        type=str,
        default="data/processed/ball/non_annotation_state.json",
        help="State file for resume/idempotent mapping",
    )
    ap.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for saved frames")
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

    if args.interactive:
        interactive_loop(
            videos=videos,
            videos_root=videos_dir,
            images_root=images_root,
            json_out=json_out,
            state_path=Path(args.state_file),
            start_game_id=int(args.start_game_id),
            frame_stride=int(args.frame_stride),
            quality=int(args.jpeg_quality),
        )
        return

    # ---- Batch mode (existing behavior) ----
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
