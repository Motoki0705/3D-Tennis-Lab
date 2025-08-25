from __future__ import annotations
from pathlib import Path
from typing import List


def discover_videos(video_dir: str | Path, exts: List[str]) -> List[Path]:
    base = Path(video_dir)
    found: List[Path] = []
    if not base.exists():
        return []
    extset = {e.lower() for e in exts}
    for p in base.iterdir():
        if p.is_file() and p.suffix.lower() in extset:
            found.append(p)
    return found


def sort_videos(paths: List[Path], mode: str) -> List[Path]:
    mode = (mode or "name").lower()
    if mode == "mtime":
        return sorted(paths, key=lambda p: p.stat().st_mtime)
    return sorted(paths, key=lambda p: p.name.lower())


def first_token(stem: str) -> str:
    s = stem.strip()
    if not s:
        return "vid"
    # split by any whitespace
    parts = s.split()
    return parts[0] if parts else s
