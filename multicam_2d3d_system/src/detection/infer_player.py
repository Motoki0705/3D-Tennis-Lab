"""Player detection inference wrapper using RT-DETR checkpoints."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import cv2
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from ..dataio import formats

try:
    from trained_models.player_analysis.rt_detr.rtdetr_loader import (  # type: ignore
        RTDetrLoadConfig,
        load_hf_rtdetr_with_ckpt,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - guard for packaging issues
    raise ImportError("RT-DETR loader not found. Ensure trained_models package is available.") from exc

_LOGGER = logging.getLogger(__name__)


def run_player_inference(
    video_paths: Iterable[Path],
    weights_path: Path,
    detector_cfg: DictConfig | dict | None = None,
) -> list[formats.FrameDetections2D]:
    """Run RT-DETR person detector on the provided videos."""

    loader_cfg, score_threshold = _build_loader_config(weights_path, detector_cfg)
    model, processor, device = load_hf_rtdetr_with_ckpt(loader_cfg)

    results: list[formats.FrameDetections2D] = []

    for video_path in video_paths:
        _LOGGER.info("Running player detector on %s", video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            _LOGGER.warning("FPS metadata missing in %s; defaulting to 30 FPS", video_path)
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_idx = 0

        with torch.inference_mode():
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                batch = processor(images=pil_image, return_tensors="pt")
                batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}

                outputs = model(**batch)
                target_size = torch.tensor([[height, width]], device=device)
                processed = processor.post_process_object_detection(
                    outputs,
                    threshold=score_threshold,
                    target_sizes=target_size,
                )[0]

                detections: list[formats.Detection2DEntry] = []
                boxes = processed.get("boxes")
                scores = processed.get("scores")
                labels = processed.get("labels")

                if boxes is not None and scores is not None and labels is not None:
                    for box, score, label in zip(boxes.cpu(), scores.cpu(), labels.cpu(), strict=False):
                        if int(label.item()) != 0:
                            continue
                        detections.append({
                            "cls": "player",
                            "bbox": tuple(float(v) for v in box.tolist()),
                            "score": float(score.item()),
                        })

                results.append({
                    "camera_id": video_path.stem,
                    "frame_idx": frame_idx,
                    "timestamp": frame_idx / fps if fps > 0 else float(frame_idx),
                    "detections": detections,
                    "image_size": (width, height),
                })

                frame_idx += 1

        cap.release()

    return results


def _build_loader_config(
    weights_path: Path,
    detector_cfg: DictConfig | dict | None,
) -> tuple[RTDetrLoadConfig, float]:
    kwargs: dict[str, object] = {
        "checkpoint_path": str(weights_path),
    }

    score_threshold = 0.5
    if detector_cfg is not None:
        if isinstance(detector_cfg, DictConfig):
            config_values_obj = OmegaConf.to_container(detector_cfg, resolve=True)
        else:
            config_values_obj = dict(detector_cfg)

        config_values = cast(dict[str, Any], config_values_obj)

        score_threshold = float(config_values.pop("score_threshold", score_threshold))
        config_values.pop("weights", None)

        allowed_fields = {
            "pretrained_model_name_or_path",
            "num_labels",
            "device",
            "strict",
            "remove_prefix",
            "allow_partial",
            "torch_compile",
        }
        for field in allowed_fields:
            if field in config_values and config_values[field] is not None:
                value = config_values[field]
                if field == "num_labels":
                    kwargs[field] = int(value)
                elif field in {"strict", "allow_partial", "torch_compile"}:
                    kwargs[field] = bool(value)
                else:
                    kwargs[field] = value

    loader_cfg = RTDetrLoadConfig(**kwargs)
    return loader_cfg, float(score_threshold)
