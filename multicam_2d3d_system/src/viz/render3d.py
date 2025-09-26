"""3D visualization helpers."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ..dataio import formats

_LOGGER = logging.getLogger(__name__)


def _collect_points(reconstruction: formats.Reconstruction3D) -> tuple[np.ndarray, list[str]]:
    objects = reconstruction.get("objects", []) or []
    points: list[tuple[float, float, float]] = []
    labels: list[str] = []
    for obj in objects:
        object_id = str(obj.get("id", "object"))
        for frame in obj.get("frames", []) or []:
            coords = frame.get("X")
            if not coords or len(coords) != 3:
                continue
            try:
                x, y, z = (float(coords[0]), float(coords[1]), float(coords[2]))
            except (TypeError, ValueError):
                continue
            points.append((x, y, z))
            labels.append(object_id)
    if not points:
        return np.empty((0, 3), dtype=float), []
    return np.asarray(points, dtype=float), labels


def _colorize(labels: Iterable[str]) -> np.ndarray:
    palette = {
        "ball": (255, 127, 14),
        "player": (44, 160, 44),
        "racket": (31, 119, 180),
    }
    colors: list[tuple[float, float, float]] = []
    for label in labels:
        base = palette.get(label, (148, 103, 189))
        colors.append(tuple(channel / 255.0 for channel in base))
    return np.asarray(colors, dtype=float)


def _write_open3d(
    points: np.ndarray,
    colors: np.ndarray,
    output_path: Path,
) -> bool:
    try:  # pragma: no cover - optional dependency
        import open3d as o3d
    except ModuleNotFoundError:
        return False

    if output_path.suffix.lower() != ".ply":
        output_path = output_path.with_suffix(".ply")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors)
    success = o3d.io.write_point_cloud(str(output_path), pcd)
    if success:
        _LOGGER.info("Wrote 3D point cloud to %s", output_path)
    else:  # pragma: no cover - defensive branch
        _LOGGER.warning("Open3D failed to write point cloud to %s", output_path)
    return success


def _write_matplotlib(
    points: np.ndarray,
    colors: np.ndarray,
    output_path: Path,
) -> bool:
    try:  # pragma: no cover - optional dependency
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _LOGGER.warning("Matplotlib not available; skipping 3D rendering")
        return False

    if output_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        output_path = output_path.with_suffix(".png")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")  # type: ignore[arg-type]
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter(xs, ys, zs, c=colors, s=8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=-60)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    _LOGGER.info("Saved 3D trajectory preview to %s", output_path)
    return True


def render_scene(
    reconstruction: formats.Reconstruction3D | None,
    output_path: Path,
    backend: str = "open3d",
) -> None:
    """Render reconstructed trajectories in 3D.

    Parameters
    ----------
    reconstruction: formats.Reconstruction3D | None
        Reconstruction payload to visualize.
    output_path: Path
        Destination file (PLY for Open3D, PNG otherwise).
    backend: str, default "open3d"
        Preferred rendering backend. Falls back to Matplotlib when unavailable.
    """

    if not reconstruction:
        _LOGGER.info("No reconstruction payload supplied; skipping 3D rendering")
        return

    points, labels = _collect_points(reconstruction)
    if points.size == 0:
        _LOGGER.info("Reconstruction does not contain 3D points; skipping rendering")
        return

    colors = _colorize(labels)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backend = (backend or "").lower()
    wrote = False
    if backend == "open3d":
        wrote = _write_open3d(points, colors, output_path)
        if not wrote:
            _LOGGER.warning("Open3D backend unavailable; falling back to Matplotlib")
            wrote = _write_matplotlib(points, colors, output_path)
    elif backend == "matplotlib":
        wrote = _write_matplotlib(points, colors, output_path)
    else:
        _LOGGER.warning("Unknown 3D backend '%s'; using Matplotlib fallback", backend)
        wrote = _write_matplotlib(points, colors, output_path)

    if not wrote:
        _LOGGER.warning("Failed to render 3D scene with any backend")
