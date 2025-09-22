"""Intrinsics estimation and loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class IntrinsicsConfig:
    """Configuration holder for intrinsic estimation."""

    model: str
    optimize_distortion: bool


class IntrinsicsSolver:
    """Loads or optimizes camera intrinsics."""

    def __init__(self, config: IntrinsicsConfig) -> None:
        self.config = config

    def solve(self, image_dir: Path) -> dict[str, Any]:
        """Return a mapping of camera_id to intrinsic parameters."""
        raise NotImplementedError("Implement intrinsics solving logic")

    def load(self, yaml_path: Path) -> dict[str, Any]:
        """Load intrinsics from a serialized YAML description."""
        raise NotImplementedError("Implement YAML loading logic")
