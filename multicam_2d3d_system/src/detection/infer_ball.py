"""Ball detection inference wrapper using the trained HRNet/TrackNetV2 stack."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from ..dataio import formats

try:  # Local import path for the loader created in trained_models
    from trained_models.ball_tracking.hrnet.hrnet_loader import (  # type: ignore
        HRNetLoadConfig,
        load_hrnet_with_ckpt,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - guard for packaging issues
    raise ImportError("HRNet loader not found. Ensure trained_models package is available.") from exc

from utils.image import get_affine_transform  # type: ignore

_LOGGER = logging.getLogger(__name__)


def run_ball_inference(
    video_paths: Iterable[Path],
    weights_path: Path,
    detector_cfg: DictConfig | dict | None = None,
) -> list[formats.FrameDetections2D]:
    """Run the HRNet ball detector on the provided videos.

    Parameters
    ----------
    video_paths:
        Ordered iterable of input videos. Each video is treated as an
        independent camera stream identified by its stem.
    weights_path:
        Filesystem path to the TrackNetV2 checkpoint (``.pth.tar``).
    detector_cfg:
        Optional configuration mapping (typically a Hydra ``DictConfig``)
        providing overrides for ``device``, ``gpu_ids``, ``config_dir``, etc.

    Returns
    -------
    list[formats.FrameDetections2D]
        One record per processed frame in each video, ordered first by video
        and then by frame index.
    """

    loader_cfg = _build_loader_config(weights_path, detector_cfg)
    detector, tracker, transform, device, resolved_cfg = load_hrnet_with_ckpt(loader_cfg)

    frames_in = int(resolved_cfg.model.frames_in)
    input_wh = (int(resolved_cfg.model.inp_width), int(resolved_cfg.model.inp_height))
    output_wh = (int(resolved_cfg.model.out_width), int(resolved_cfg.model.out_height))
    out_scales = tuple(int(scale) for scale in resolved_cfg.model.out_scales)

    torch_device = torch.device(device)
    results: list[formats.FrameDetections2D] = []

    for video_path in video_paths:
        _LOGGER.info("Running ball detector on %s", video_path)
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

        frame_buffer: deque[torch.Tensor] = deque(maxlen=frames_in)
        tracker.refresh()

        frame_idx = 0
        trans_input = _compute_input_transform((height, width), input_wh)
        trans_outputs_template = _compute_output_transforms((height, width), output_wh, out_scales, torch_device)

        with torch.inference_mode():
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                warped = cv2.warpAffine(frame_bgr, trans_input, input_wh, flags=cv2.INTER_LINEAR)
                pil_img = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
                tensor = transform(pil_img)
                frame_buffer.append(tensor)

                detections: list[formats.Detection2DEntry] = []

                if len(frame_buffer) >= frames_in:
                    clip_tensor = torch.cat(list(frame_buffer), dim=0).unsqueeze(0).to(torch_device)
                    trans_outputs = {scale: mat.clone() for scale, mat in trans_outputs_template.items()}

                    batch_results, _ = detector.run_tensor(clip_tensor, trans_outputs)
                    preds = batch_results[0][frames_in - 1]
                    tracked = tracker.update(preds)

                    if tracked.get("visi"):
                        detections.append({
                            "cls": "ball",
                            "point": (float(tracked["x"]), float(tracked["y"])),
                            "score": float(tracked.get("score", 0.0)),
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
) -> HRNetLoadConfig:
    kwargs: dict[str, object] = {
        "checkpoint_path": str(weights_path),
    }

    if detector_cfg is not None:
        if isinstance(detector_cfg, DictConfig):
            config_values_obj = OmegaConf.to_container(detector_cfg, resolve=True)
        else:
            config_values_obj = dict(detector_cfg)

        config_values = cast(dict[str, Any], config_values_obj)

        config_dir = config_values.get("config_dir")
        if config_dir:
            kwargs["config_dir"] = str(config_dir)

        config_name = config_values.get("config_name")
        if config_name:
            kwargs["config_name"] = str(config_name)

        overrides = config_values.get("overrides")
        if overrides:
            if isinstance(overrides, list | tuple):
                kwargs["overrides"] = tuple(str(item) for item in overrides)
            else:
                kwargs["overrides"] = (str(overrides),)

        device = config_values.get("device")
        if device:
            kwargs["device"] = str(device)

        gpu_ids = config_values.get("gpu_ids")
        if gpu_ids:
            if isinstance(gpu_ids, list | tuple):
                kwargs["gpu_ids"] = tuple(int(idx) for idx in gpu_ids)
            else:
                kwargs["gpu_ids"] = (int(gpu_ids),)

    return HRNetLoadConfig(**kwargs)


def _compute_input_transform(img_shape: tuple[int, int], input_wh: tuple[int, int]) -> np.ndarray:
    height, width = img_shape
    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    scale = float(max(height, width))
    return get_affine_transform(center, scale, 0, list(map(int, input_wh)), inv=0)


def _compute_output_transforms(
    img_shape: tuple[int, int],
    output_wh: tuple[int, int],
    out_scales: Iterable[int],
    device: torch.device,
) -> dict[int, torch.Tensor]:
    height, width = img_shape
    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    base_scale = float(max(height, width))

    transforms: dict[int, torch.Tensor] = {}
    out_w, out_h = map(int, output_wh)
    for level in out_scales:
        mat = get_affine_transform(center, base_scale, 0, [out_w, out_h], inv=1)
        transforms[int(level)] = torch.tensor(mat, dtype=torch.float32, device=device).unsqueeze(0)
        out_w = max(out_w // 2, 1)
        out_h = max(out_h // 2, 1)
    return transforms
