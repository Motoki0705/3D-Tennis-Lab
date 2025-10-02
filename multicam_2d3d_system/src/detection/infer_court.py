"""Court layout detection wrapper using the DINO-FPN heatmap model."""

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
    from trained_models.court_pose.dino_fpn.dino_fpn_loader import (  # type: ignore
        CoatLoadConfig,
        load_coat_with_ckpt,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - guard for packaging issues
    raise ImportError("Court pose loader not found. Ensure trained_models package is available.") from exc

_LOGGER = logging.getLogger(__name__)


def run_court_inference(
    video_paths: Iterable[Path],
    weights_path: Path,
    detector_cfg: DictConfig | dict | None = None,
) -> list[formats.FrameDetections2D]:
    """Run DINO-FPN heatmap model to extract court keypoints per frame."""

    loader_cfg, score_threshold, max_keypoints = _build_loader_config(weights_path, detector_cfg)
    model, transform, device = load_coat_with_ckpt(loader_cfg)

    batch_size = 1
    use_half = False
    if detector_cfg is not None:
        cfg_obj = detector_cfg if isinstance(detector_cfg, DictConfig) else OmegaConf.create(detector_cfg)
        batch_size = int(getattr(cfg_obj, "batch_size", 1) or 1)
        use_half = bool(getattr(cfg_obj, "use_half", False))

    if batch_size <= 0:
        batch_size = 1

    torch_device = torch.device(device)
    if use_half and torch_device.type != "cuda":
        _LOGGER.warning(
            "Half precision requested for court detector but device is %s; using float32 instead",
            torch_device,
        )
        use_half = False

    if use_half:
        try:
            model = model.half()  # type: ignore[assignment]
        except AttributeError:  # pragma: no cover - defensive fallback
            _LOGGER.warning("Court detector does not support half precision; continuing in float32")
            use_half = False

    results: list[formats.FrameDetections2D] = []

    def _flush_batch(
        tensors: list[torch.Tensor],
        metas: list[dict[str, Any]],
    ) -> None:
        if not tensors:
            return

        batch_tensor = torch.stack(tensors, dim=0)
        if use_half and batch_tensor.dtype == torch.float32:
            batch_tensor = batch_tensor.half()
        batch_tensor = batch_tensor.to(torch_device)

        outputs = model(batch_tensor)
        heatmaps_batch = outputs[0] if isinstance(outputs, tuple | list) else outputs
        heatmaps_batch = heatmaps_batch.detach().cpu()

        for meta, heatmaps in zip(metas, heatmaps_batch, strict=False):
            num_keypoints, hm_height, hm_width = heatmaps.shape

            width = meta["width"]
            height = meta["height"]

            scale_x = width / hm_width if hm_width else 1.0
            scale_y = height / hm_height if hm_height else 1.0

            keypoint_detections: list[formats.Detection2DEntry] = []
            for channel in range(num_keypoints):
                hm = heatmaps[channel]
                score_value, flat_idx = torch.max(hm.view(-1), dim=0)
                score_float = float(score_value.item())
                if score_float < score_threshold:
                    continue

                y = int(flat_idx // hm_width)
                x = int(flat_idx % hm_width)

                x_orig = float(x * scale_x)
                y_orig = float(y * scale_y)

                keypoint_detections.append({
                    "cls": "court",
                    "point": (x_orig, y_orig),
                    "score": score_float,
                })

            if max_keypoints is not None and len(keypoint_detections) > max_keypoints:
                keypoint_detections.sort(key=lambda det: det.get("score", 0.0), reverse=True)
                keypoint_detections = keypoint_detections[:max_keypoints]

            record = meta["record"]
            record["detections"] = keypoint_detections

        tensors.clear()
        metas.clear()

    for video_path in video_paths:
        _LOGGER.info("Running court detector on %s", video_path)
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
        batch_tensors: list[torch.Tensor] = []
        batch_meta: list[dict[str, Any]] = []

        with torch.inference_mode():
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                tensor = transform(pil_image)
                if tensor.ndim != 3:
                    raise ValueError("Court transform must return a CHW tensor")

                record: formats.FrameDetections2D = {
                    "camera_id": video_path.stem,
                    "frame_idx": frame_idx,
                    "timestamp": frame_idx / fps if fps > 0 else float(frame_idx),
                    "detections": [],
                    "image_size": (width, height),
                }
                results.append(record)

                batch_tensors.append(tensor)
                batch_meta.append({
                    "record": record,
                    "width": width,
                    "height": height,
                })

                if len(batch_tensors) >= batch_size:
                    _flush_batch(batch_tensors, batch_meta)

                frame_idx += 1

            _flush_batch(batch_tensors, batch_meta)

        cap.release()

    return results


def _build_loader_config(
    weights_path: Path,
    detector_cfg: DictConfig | dict | None,
) -> tuple[CoatLoadConfig, float, int | None]:
    kwargs: dict[str, Any] = {
        "checkpoint_path": str(weights_path),
    }

    score_threshold = 0.3
    max_keypoints: int | None = None

    if detector_cfg is not None:
        if isinstance(detector_cfg, DictConfig):
            config_values_obj = OmegaConf.to_container(detector_cfg, resolve=True)
        else:
            config_values_obj = dict(detector_cfg)

        config_values = cast(dict[str, Any], config_values_obj)

        score_threshold = float(config_values.pop("score_threshold", score_threshold))
        max_k = config_values.pop("max_keypoints", None)
        if max_k is not None:
            max_keypoints = int(max_k)

        config_values.pop("weights", None)
        config_values.pop("batch_size", None)
        config_values.pop("use_half", None)

        allowed_fields = {
            "heatmap_channels",
            "decoder_channels",
            "backbone_name",
            "weights_path",
            "pad_to_multiple",
            "device",
            "strict",
            "remove_prefix",
            "allow_partial",
            "resize_long_side",
        }
        for field in allowed_fields:
            if field in config_values and config_values[field] is not None:
                value = config_values[field]
                if field == "decoder_channels":
                    kwargs[field] = [int(v) for v in value]
                elif field == "heatmap_channels":
                    kwargs[field] = int(value)
                elif field in {"pad_to_multiple", "resize_long_side"}:
                    kwargs[field] = None if value is None else int(value)
                elif field in {"strict", "allow_partial"}:
                    kwargs[field] = bool(value)
                else:
                    kwargs[field] = value

    loader_cfg = CoatLoadConfig(**kwargs)
    return loader_cfg, float(score_threshold), max_keypoints
