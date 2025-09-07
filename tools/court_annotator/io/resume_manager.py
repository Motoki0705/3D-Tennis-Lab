from __future__ import annotations
from pathlib import Path
from typing import Dict
import json
from datetime import datetime, timezone


SCHEMA = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resume_path(out_dir: Path) -> Path:
    return out_dir / "resume.json"


def load_resume(out_dir: Path) -> Dict:
    p = resume_path(out_dir)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Backfill defaults
        data.setdefault("schema", SCHEMA)
        data.setdefault("last_scanned", _now_iso())
        data.setdefault("videos", {})
        data.setdefault("image_id_map", {})
        data.setdefault("next_image_id", 1)
        return data
    return {
        "schema": SCHEMA,
        "last_scanned": _now_iso(),
        "videos": {},
        "image_id_map": {},
        "next_image_id": 1,
    }


def save_resume(out_dir: Path, data: Dict):
    p = resume_path(out_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_video(resume: Dict, video_path: str) -> int:
    videos = resume.setdefault("videos", {})
    if video_path in videos:
        return int(videos[video_path]["video_id"])
    vid = max([v["video_id"] for v in videos.values()] or [0]) + 1
    videos[video_path] = {"video_id": vid, "next_frame": 0, "done": False}
    return vid


def get_video_id(resume: Dict, video_path: str) -> int:
    return (
        int(resume["videos"][video_path]["video_id"])
        if video_path in resume.get("videos", {})
        else ensure_video(resume, video_path)
    )


def frame_key(video_path: str, frame_idx: int) -> str:
    return f"{video_path}#{frame_idx:06d}"


def alloc_image_id(resume: Dict, video_path: str, frame_idx: int) -> int:
    key = frame_key(video_path, frame_idx)
    m = resume.setdefault("image_id_map", {})
    if key in m:
        return int(m[key])
    nid = int(resume.setdefault("next_image_id", 1))
    m[key] = nid
    resume["next_image_id"] = nid + 1
    return nid


def update_video_progress(resume: Dict, video_path: str, next_frame: int | None = None, done: bool | None = None):
    v = resume.setdefault("videos", {}).setdefault(
        video_path, {"video_id": ensure_video(resume, video_path), "next_frame": 0, "done": False}
    )
    if next_frame is not None:
        v["next_frame"] = int(next_frame)
    if done is not None:
        v["done"] = bool(done)
    resume["last_scanned"] = _now_iso()
