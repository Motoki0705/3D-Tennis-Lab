"""3D visualization helpers."""

from __future__ import annotations

from pathlib import Path

from ..dataio import formats


def render_scene(
    reconstruction: formats.Reconstruction3D,
    output_path: Path,
    backend: str = "open3d",
) -> None:
    """Render reconstructed trajectories in 3D."""
    raise NotImplementedError("Implement 3D rendering")
