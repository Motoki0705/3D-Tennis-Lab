"""Data serialization helpers for detections, tracks, and 3D reconstructions."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

from . import formats

_LOGGER = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_jsonl(records: Iterable[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle)
            handle.write("\n")


def write_detections(
    detections: list[formats.FrameDetections2D],
    output_dir: Path,
) -> None:
    """Persist 2D detections as JSONL."""
    output_dir = Path(output_dir)
    _ensure_dir(output_dir)
    output_path = output_dir / "detections.jsonl"
    _write_jsonl(detections, output_path)
    _LOGGER.info("Wrote %d detection frames to %s", len(detections), output_path)


def write_tracks(tracks: list[formats.CameraTracks], output_dir: Path) -> None:
    """Persist 2D track bundles as JSONL."""
    output_dir = Path(output_dir)
    _ensure_dir(output_dir)
    output_path = output_dir / "tracks.jsonl"
    _write_jsonl(tracks, output_path)
    _LOGGER.info("Wrote %d track bundles to %s", len(tracks), output_path)


def write_reconstruction(
    reconstruction: formats.Reconstruction3D,
    output_path: Path,
) -> None:
    """Persist 3D reconstruction results.

    Parquet export is attempted when the suffix is `.parquet`. If pandas/pyarrow are
    unavailable the routine falls back to JSON output.
    """
    output_path = Path(output_path)
    _ensure_dir(output_path.parent)

    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        try:  # pragma: no cover - optional dependency
            import pandas as pd  # type: ignore
        except ModuleNotFoundError:  # pragma: no cover - optional dependency
            fallback = output_path.with_suffix(".json")
            _LOGGER.warning(
                "pandas/pyarrow unavailable; writing reconstruction JSON to %s instead of %s",
                fallback,
                output_path,
            )
            with fallback.open("w", encoding="utf-8") as handle:
                json.dump(reconstruction, handle, ensure_ascii=False, indent=2)
            return

        records = []
        for obj in reconstruction.get("objects", []):
            object_id = obj.get("id")
            for frame in obj.get("frames", []):
                record = {"object_id": object_id, **frame}
                records.append(record)
        pd.DataFrame.from_records(records).to_parquet(output_path)  # type: ignore[attr-defined]
        _LOGGER.info("Wrote reconstruction with %d frames to %s", len(records), output_path)
        return

    if suffix in {".jsonl"}:
        _write_jsonl(reconstruction.get("objects", []), output_path)
    else:
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(reconstruction, handle, ensure_ascii=False, indent=2)
    _LOGGER.info("Wrote reconstruction payload to %s", output_path)
