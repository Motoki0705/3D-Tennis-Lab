"""Hydra pipeline for multi-camera 2D detection."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from pathlib import Path

from omegaconf import DictConfig

from ..dataio import writers
from ..detection import infer_ball, infer_player

_LOGGER = logging.getLogger(__name__)

DetectorFn = Callable[[Iterable[Path], DictConfig], list]


def _run_detector(name: str, fn: DetectorFn, *args) -> list:
    try:
        results = fn(*args)
    except NotImplementedError:
        _LOGGER.info("Detector '%s' not implemented yet; skipping", name)
        return []
    except Exception:  # pragma: no cover - propagate unexpected errors
        _LOGGER.exception("Detector '%s' failed", name)
        raise
    _LOGGER.info("Detector '%s' produced %d frames", name, len(results))
    return results


def run(cfg: DictConfig) -> None:
    """Main entry point for the 2D detection stage."""
    videos_root = Path(cfg.data.root)
    video_ext = cfg.data.frames.get("format", "mp4")
    videos = [videos_root / f"{camera_id}.{video_ext}" for camera_id in cfg.data.cameras]

    detections_dir = Path(cfg.data.annotations.detections_dir)

    player_cfg = cfg.detection.player
    ball_cfg = cfg.detection.ball

    combined_results: list = []
    combined_results.extend(_run_detector("player", infer_player.run_player_inference, videos, player_cfg))
    combined_results.extend(
        _run_detector(
            "ball",
            infer_ball.run_ball_inference,
            videos,
            ball_cfg,
        )
    )

    if "court" in cfg.detection:
        from ..detection import infer_court  # Local import until implementation lands

        court_cfg = cfg.detection.court
        combined_results.extend(
            _run_detector(
                "court",
                infer_court.run_court_inference,
                videos,
                court_cfg,
            )
        )

    if not combined_results:
        _LOGGER.warning("No detections generated; skipping serialization")
        return

    writers.write_detections(combined_results, detections_dir)
