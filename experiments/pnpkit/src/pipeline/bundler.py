from __future__ import annotations

import os
import re
from typing import Dict, List, Callable

from .base import FrameObs


def by_prefix(frames: List[FrameObs], pattern: str) -> Dict[str, List[FrameObs]]:
    """Group frames by regex prefix match on basename.

    pattern examples:
    - r"^[^_]+_"  -> leading token until underscore (e.g., "Dimitrov_")
    - r"^[^-]+-"  -> leading token until dash
    """
    rgx = re.compile(pattern)
    buckets: Dict[str, List[FrameObs]] = {}
    for fr in frames:
        bn = os.path.basename(fr.image_path)
        m = rgx.search(bn)
        key = m.group(0) if m else ""
        buckets.setdefault(key, []).append(fr)
    for k in buckets:
        buckets[k].sort(key=lambda f: f.frame_idx)
    return buckets


def by_meta(frames: List[FrameObs], field: str = "camera_id") -> Dict[str, List[FrameObs]]:
    buckets: Dict[str, List[FrameObs]] = {}
    for fr in frames:
        cam = None
        if hasattr(fr, "meta") and isinstance(getattr(fr, "meta"), dict):
            cam = fr.meta.get(field)  # type: ignore[attr-defined]
        key = str(cam) if cam is not None else ""
        buckets.setdefault(key, []).append(fr)
    for k in buckets:
        buckets[k].sort(key=lambda f: f.frame_idx)
    return buckets


def by_fn(frames: List[FrameObs], fn: Callable[[FrameObs], str]) -> Dict[str, List[FrameObs]]:
    buckets: Dict[str, List[FrameObs]] = {}
    for fr in frames:
        key = str(fn(fr))
        buckets.setdefault(key, []).append(fr)
    for k in buckets:
        buckets[k].sort(key=lambda f: f.frame_idx)
    return buckets


def group_frames(frames: List[FrameObs], strategy: str, **params) -> Dict[str, List[FrameObs]]:
    if strategy == "by_prefix":
        patt = params.get("pattern") or r"^[^_]+_"
        return by_prefix(frames, patt)
    if strategy == "by_meta":
        field = params.get("field") or "camera_id"
        return by_meta(frames, field)
    if strategy == "by_fn":
        fn = params.get("fn")
        if not callable(fn):
            raise ValueError("bundler.by_fn requires a callable 'fn'")
        return by_fn(frames, fn)
    raise ValueError(f"Unknown bundling strategy: {strategy}")
