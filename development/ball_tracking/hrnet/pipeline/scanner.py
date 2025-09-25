from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .state import StateRepository

SUPPORTED_EXT = {".mp4", ".mov", ".mkv", ".avi"}


def iter_videos(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXT:
            yield path


def scan_new_videos(root: Path, repo: StateRepository) -> list[str]:
    """Scan the input directory and register newly discovered videos."""

    root = root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    registered: list[str] = []
    for video_path in iter_videos(root):
        video_id = repo.register_video(video_path)
        registered.append(video_id)
    return registered
