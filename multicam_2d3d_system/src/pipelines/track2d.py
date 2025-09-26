"""Hydra pipeline for per-camera tracking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from ..dataio import readers, writers
from ..detection import tracking
from ..viz import overlay2d

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

    viz_cfg: Any = getattr(cfg.export, "visualization", None)
    overlay_cfg: Any = getattr(viz_cfg, "overlay_video", None)
    if overlay_cfg and getattr(overlay_cfg, "enabled", False):
        videos_root = Path(cfg.data.root)
        video_ext = cfg.data.frames.get("format", "mp4")
        video_sources = {str(camera_id): videos_root / f"{camera_id}.{video_ext}" for camera_id in cfg.data.cameras}
        output_dir = Path(getattr(overlay_cfg, "output_dir", Path(cfg.workspace.outputs) / "viz" / "overlays"))
        codec = getattr(overlay_cfg, "codec", "mp4v")
        overlay2d.render_overlays(
            detections,
            output_dir,
            video_sources=video_sources,
            tracks=tracks,
            codec=codec,
        )
