"""Extrinsics estimation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExtrinsicsConfig:
    """Configuration for extrinsic parameter solvers."""

    method: str
    use_court_constraints: bool = True


class ExtrinsicsSolver:
    """Solves for camera poses relative to the court frame."""

    def __init__(self, config: ExtrinsicsConfig) -> None:
        self.config = config

    def solve(self, detections_path: Path) -> dict[str, Any]:
        """Return a mapping of camera_id to 4x4 transforms."""
        raise NotImplementedError("Implement extrinsics solving logic")

    def refine(self, current_solution: dict[str, Any]) -> dict[str, Any]:
        """Refine extrinsics using additional correspondences."""
        raise NotImplementedError("Implement extrinsics refinement")
