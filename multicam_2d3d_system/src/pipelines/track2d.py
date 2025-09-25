"""Hydra pipeline for per-camera tracking."""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig

from ..dataio import readers, writers
from ..detection import tracking

_LOGGER = logging.getLogger(__name__)


def run(cfg: DictConfig) -> None:
    """Execute the 2D tracking stage."""
    detection_dir = Path(cfg.data.annotations.detections_dir)
    detections = readers.load_detections([detection_dir])

    if not detections:
        _LOGGER.warning("No detections available at %s; skipping tracking", detection_dir)
        return

    tracker_cfg = cfg.detection.tracker
    try:
        tracks = tracking.track_detections(
            detections,
            method=tracker_cfg.method,
            max_frame_gap=getattr(tracker_cfg, "max_frame_gap", 5),
            ball_distance_px=getattr(tracker_cfg, "ball_distance_px", 50.0),
            player_distance_px=getattr(tracker_cfg, "player_distance_px", 120.0),
            min_track_length=getattr(tracker_cfg, "min_track_length", 3),
        )
    except NotImplementedError:
        _LOGGER.info("Tracker '%s' not implemented; skipping", tracker_cfg.method)
        return

    writers.write_tracks(tracks, Path(cfg.data.annotations.tracks_dir))
