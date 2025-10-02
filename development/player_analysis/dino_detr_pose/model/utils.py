"""Utilities for accessing third_party DETRPose modules."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def ensure_detrpose_imports() -> None:
    """Append the DETRPose src directory to sys.path once."""
    root = Path(__file__).resolve().parents[4]
    src = root / "third_party" / "DETRPose" / "src"
    if not src.exists():
        raise FileNotFoundError(f"DETRPose source directory not found at {src}")
    sys.path.append(str(src))


__all__ = ["ensure_detrpose_imports"]
