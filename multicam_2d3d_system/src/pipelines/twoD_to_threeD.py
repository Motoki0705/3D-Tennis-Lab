"""Hydra pipeline that converts 2D tracks into 3D trajectories."""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig

from ..dataio import readers, writers
from ..geometry import court_frame, temporal_opt

_LOGGER = logging.getLogger(__name__)


def _build_output_path(cfg: DictConfig) -> Path:
    export_cfg = cfg.export.reconstruction
    output_root = Path(export_cfg.path)
    output_root.mkdir(parents=True, exist_ok=True)
    extension = export_cfg.format.lower()
    if extension.startswith("."):
        extension = extension.lstrip(".")
    return output_root / f"reconstruction.{extension}"


def run(cfg: DictConfig) -> None:
    """Execute the multi-stage 2D→3D reconstruction flow."""
    tracks_dir = Path(cfg.data.annotations.tracks_dir)
    track_bundles = readers.load_tracks([tracks_dir])

    if not track_bundles:
        _LOGGER.warning("No tracks found in %s; skipping 3D reconstruction", tracks_dir)
        empty_reconstruction: dict = {"objects": []}
        writers.write_reconstruction(empty_reconstruction, _build_output_path(cfg))
        return

    _LOGGER.info("Loaded %d camera track bundles", len(track_bundles))

    # Placeholder reconstruction container until triangulation is implemented.
    reconstruction: dict = {"objects": []}

    try:
        smoothed_objects: list = temporal_opt.smooth_trajectories(
            reconstruction.get("objects", []),
            method=cfg.triangulation.smoothing.filter,
        )
        reconstruction["objects"] = smoothed_objects
    except NotImplementedError:
        _LOGGER.info("Temporal smoothing not implemented; continuing with raw objects")

    calibration_dir = Path(cfg.workspace.outputs) / "calibration"
    calibration_entries = readers.load_calibration(calibration_dir / "cameras.yaml")
    if calibration_entries:
        calibration_map = {entry["camera_id"]: entry for entry in calibration_entries if "camera_id" in entry}
        try:
            reconstruction = court_frame.to_court_frame(reconstruction, calibration_map)
        except NotImplementedError:
            _LOGGER.info("Court frame alignment not implemented; exporting raw coordinates")
    else:
        _LOGGER.info("Skipping court frame alignment; no calibration entries available")

    writers.write_reconstruction(reconstruction, _build_output_path(cfg))
