"""Camera synchronization primitives."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class SyncConfig:
    """Parameters controlling timestamp alignment."""

    reference_camera: str
    max_offset_ms: float
    max_drift_ppm: float


def estimate_offsets(metadata: Iterable[dict[str, float]], config: SyncConfig) -> dict[str, float]:
    """Compute per-camera timestamp offsets relative to the reference camera."""
    raise NotImplementedError("Implement synchronization logic")
