"""Hydra pipeline that converts 2D tracks into 3D trajectories."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from omegaconf import DictConfig

from ..dataio import formats, readers, writers
from ..geometry import court_frame, temporal_opt, triangulation

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
    detections_dir = Path(cfg.data.annotations.detections_dir)
    detections = readers.load_detections([detections_dir])
    if not detections:
        _LOGGER.warning("No detections available at %s; cannot produce 3D reconstruction", detections_dir)
        empty_reconstruction: dict = {"objects": []}
        writers.write_reconstruction(empty_reconstruction, _build_output_path(cfg))
        return

    calibration_dir = Path(cfg.workspace.outputs) / "calibration"
    calibration_entries = readers.load_calibration(calibration_dir / "cameras.yaml")
    calibration_map = {entry["camera_id"]: entry for entry in calibration_entries if "camera_id" in entry}
    if len(calibration_map) < 2:
        _LOGGER.warning("Need calibration for at least two cameras; found %d", len(calibration_map))

    frame_buckets: dict[tuple[int, int], list[tuple[str, formats.Detection2DEntry]]] = defaultdict(list)
    timestamps: dict[tuple[int, int], float] = {}

    for record in detections:
        camera_id = str(record["camera_id"])
        frame_idx = int(record.get("frame_idx", 0))
        timestamp = float(record.get("timestamp", frame_idx))
        bucket_key = (frame_idx, round(timestamp * 1000))
        for det in record.get("detections", []):
            if det.get("cls") != "ball":
                continue
            if det.get("point") is None and det.get("bbox") is None:
                continue
            frame_buckets[bucket_key].append((camera_id, det))
            timestamps[bucket_key] = timestamp

    ball_frames: list[formats.Frame3DEntry] = []

    for key, observations in sorted(frame_buckets.items(), key=lambda kv: kv[0]):
        if len(observations) < 2:
            continue
        try:
            point3d = triangulation.triangulate(observations, calibration_map)
        except Exception as exc:  # pragma: no cover - guard against singular configurations
            _LOGGER.debug("Triangulation failed for frame %s: %s", key, exc)
            continue
        timestamp = timestamps.get(key, float(key[0]))
        ball_frames.append({
            "timestamp": timestamp,
            "X": point3d,
            "src": {
                "frame_idx": key[0],
                "cameras": [camera_id for camera_id, _ in observations],
            },
        })

    reconstruction: dict = {
        "objects": [
            {
                "id": "ball",
                "frames": ball_frames,
            }
        ]
        if ball_frames
        else [],
    }

    smoothing_cfg = cfg.triangulation.smoothing
    window = getattr(smoothing_cfg, "window", None)
    kalman_cfg = None
    if hasattr(smoothing_cfg, "get"):
        kalman_cfg = smoothing_cfg.get("kalman")
    if window is None and kalman_cfg:
        window = kalman_cfg.get("smoothing_window", 5)
    if window is None:
        window = 5

    try:
        reconstruction_objects = temporal_opt.smooth_trajectories(
            reconstruction.get("objects", []),
            method=smoothing_cfg.filter,
            window=int(window),
        )
        reconstruction["objects"] = reconstruction_objects
    except NotImplementedError:
        _LOGGER.info("Temporal smoothing method not implemented; exporting raw trajectories")

    if calibration_map:
        reconstruction = court_frame.to_court_frame(reconstruction, calibration_map)
    else:
        _LOGGER.info("Skipping court frame alignment; no calibration entries available")

    writers.write_reconstruction(reconstruction, _build_output_path(cfg))
