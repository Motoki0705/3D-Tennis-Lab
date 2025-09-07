from __future__ import annotations
from pathlib import Path
import json


SCHEMA = 1


def _load_meta(meta_path: Path) -> dict:
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"schema": SCHEMA, "videos": []}


def update_video_meta(out_dir: Path, video_id: int, video_path: str, fps: float, nframes: int, width: int, height: int):
    meta_path = out_dir / "meta.json"
    data = _load_meta(meta_path)
    vids = data.setdefault("videos", [])
    # find existing
    idx = None
    for i, v in enumerate(vids):
        if int(v.get("video_id")) == int(video_id):
            idx = i
            break
    entry = {
        "video_id": int(video_id),
        "path": video_path,
        "stem": Path(video_path).stem,
        "fps": float(fps),
        "nframes": int(nframes),
        "width": int(width),
        "height": int(height),
    }
    if idx is not None:
        vids[idx] = entry
    else:
        vids.append(entry)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
