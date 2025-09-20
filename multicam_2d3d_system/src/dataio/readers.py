"""Data loading helpers for detections, tracks, and calibration artifacts."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

from omegaconf import OmegaConf

from . import formats

_LOGGER = logging.getLogger(__name__)


def _collect_files(paths: Iterable[Path], suffixes: Sequence[str]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            for suffix in suffixes:
                files.extend(sorted(path.glob(f"*{suffix}")))
        elif path.suffix.lower() in suffixes:
            files.append(path)
    return files


def load_detections(paths: Iterable[Path]) -> list[formats.FrameDetections2D]:
    """Load per-frame 2D detections from JSON/JSONL files."""
    files = _collect_files(paths, (".jsonl", ".json"))
    detections: list[formats.FrameDetections2D] = []
    for file_path in files:
        if file_path.suffix.lower() == ".jsonl":
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    detections.append(json.loads(line))
        else:
            with file_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                detections.extend(data)
            else:
                detections.append(data)
    if detections:
        _LOGGER.info("Loaded %d 2D detection frames from %d files", len(detections), len(files))
    else:
        _LOGGER.info("No detections found in %s", [str(p) for p in paths])
    return detections


def load_tracks(paths: Iterable[Path]) -> list[formats.CameraTracks]:
    """Load per-camera track bundles from JSON/JSONL files."""
    files = _collect_files(paths, (".jsonl", ".json"))
    track_bundles: list[formats.CameraTracks] = []
    for file_path in files:
        if file_path.suffix.lower() == ".jsonl":
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    track_bundles.append(json.loads(line))
        else:
            with file_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                track_bundles.extend(data)
            else:
                track_bundles.append(data)
    if track_bundles:
        _LOGGER.info("Loaded %d track bundles from %d files", len(track_bundles), len(files))
    else:
        _LOGGER.info("No track bundles found in %s", [str(p) for p in paths])
    return track_bundles


def load_calibration(path: Path) -> list[formats.CalibrationEntry]:
    """Load camera calibration artifacts from YAML or JSON."""
    path = Path(path)
    if path.is_dir():
        candidates = sorted(path.glob("*.yaml")) + sorted(path.glob("*.yml"))
        if not candidates:
            _LOGGER.info("No calibration files found in %s", path)
            return []
        path = candidates[0]

    if not path.exists():
        _LOGGER.info("Calibration file %s does not exist", path)
        return []

    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if isinstance(cfg, list):
        entries = cfg
    elif isinstance(cfg, dict):
        if "cameras" in cfg and isinstance(cfg["cameras"], list):
            entries = cfg["cameras"]
        else:
            entries = [
                {"camera_id": camera_id, **payload} for camera_id, payload in cfg.items() if isinstance(payload, dict)
            ]
    else:  # pragma: no cover - defensive fallback
        raise TypeError(f"Unsupported calibration format in {path}")

    _LOGGER.info("Loaded calibration for %d cameras from %s", len(entries), path)
    return entries  # type: ignore[return-value]
