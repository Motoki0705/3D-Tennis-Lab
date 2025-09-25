"""Player pose estimation wrapper using ViT-Pose checkpoints."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from ..dataio import formats

try:
    from trained_models.player_analysis.vit_pose.vit_pose_loader import (  # type: ignore
        PoseLoadConfig,
        load_pose_from_hub,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - guard for packaging issues
    raise ImportError("ViT-Pose loader not found. Ensure trained_models package is available.") from exc

_LOGGER = logging.getLogger(__name__)


def run_pose_inference(
    video_paths: Iterable[Path],
    weights_path: Path,
    detector_cfg: DictConfig | dict | None = None,
    player_detections: list[formats.FrameDetections2D] | None = None,
) -> list[formats.FrameDetections2D]:
    """Run ViT-Pose on player detections to produce per-frame keypoints."""

    if player_detections is None:
        raise ValueError("Player detections are required for pose inference.")

    loader_cfg, keypoint_threshold = _build_loader_config(weights_path, detector_cfg)
    pose_model, pose_processor, pose_device = load_pose_from_hub(loader_cfg)

    detection_index = _index_player_detections(player_detections)
    results: list[formats.FrameDetections2D] = []

    for video_path in video_paths:
        _LOGGER.info("Running pose estimator on %s", video_path)
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

                frame_key = (video_path.stem, frame_idx)
                player_boxes = detection_index.get(frame_key, [])

                detections: list[formats.Detection2DEntry] = []

                if player_boxes:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)

                    boxes_xyxy = np.array([box_info["bbox"] for box_info in player_boxes], dtype=np.float32)
                    boxes_xywh = boxes_xyxy.copy()
                    boxes_xywh[:, 2] -= boxes_xywh[:, 0]
                    boxes_xywh[:, 3] -= boxes_xywh[:, 1]

                    pose_inputs = pose_processor(
                        pil_image,
                        boxes=[boxes_xywh],
                        return_tensors="pt",
                    )
                    pose_inputs = {
                        key: value.to(pose_device) if hasattr(value, "to") else value
                        for key, value in pose_inputs.items()
                    }

                    pose_outputs = pose_model(**pose_inputs)
                    pose_results = pose_processor.post_process_pose_estimation(
                        pose_outputs,
                        boxes=[boxes_xywh],
                    )[0]

                    for idx, pose_entry in enumerate(pose_results):
                        keypoints = pose_entry.get("keypoints", [])
                        kp_scores = pose_entry.get("scores", [])

                        filtered_keypoints = []
                        for kp_idx, point in enumerate(keypoints):
                            x_coord, y_coord = float(point[0]), float(point[1])
                            score_val = float(kp_scores[kp_idx]) if kp_idx < len(kp_scores) else 0.0
                            if score_val < keypoint_threshold:
                                continue
                            filtered_keypoints.append({
                                "x": x_coord,
                                "y": y_coord,
                                "score": score_val,
                            })

                        if not filtered_keypoints:
                            continue

                        bbox_xyxy = boxes_xyxy[idx].tolist()
                        base_score = float(player_boxes[idx].get("score", 0.0))
                        avg_pose_score = float(np.mean([kp["score"] for kp in filtered_keypoints]))

                        detections.append({
                            "cls": "pose",
                            "bbox": tuple(float(v) for v in bbox_xyxy),
                            "score": float(max(base_score, avg_pose_score)),
                            "keypoints": filtered_keypoints,
                        })

                results.append({
                    "camera_id": frame_key[0],
                    "frame_idx": frame_idx,
                    "timestamp": frame_idx / fps if fps > 0 else float(frame_idx),
                    "detections": detections,
                    "image_size": (width, height),
                })

                frame_idx += 1

        cap.release()

    return results


def _index_player_detections(
    player_detections: list[formats.FrameDetections2D],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    index: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in player_detections:
        key = (record["camera_id"], int(record["frame_idx"]))
        detections = []
        for det in record.get("detections", []):
            if det.get("cls") != "player":
                continue
            bbox = det.get("bbox")
            if bbox is None:
                continue
            detections.append({
                "bbox": tuple(float(v) for v in bbox),
                "score": float(det.get("score", 0.0)),
            })
        if detections:
            index[key] = detections
    return index


def _build_loader_config(
    weights_path: Path,
    detector_cfg: DictConfig | dict | None,
) -> tuple[PoseLoadConfig, float]:
    kwargs: dict[str, Any] = {}

    keypoint_threshold = 0.3

    if detector_cfg is not None:
        if isinstance(detector_cfg, DictConfig):
            config_values_obj = OmegaConf.to_container(detector_cfg, resolve=True)
        else:
            config_values_obj = dict(detector_cfg)

        config_values = cast(dict[str, Any], config_values_obj)

        keypoint_threshold = float(config_values.pop("keypoint_threshold", keypoint_threshold))
        config_values.pop("weights", None)

        allowed_fields = {
            "model_id",
            "device",
            "dtype",
            "use_device_map",
            "cache_dir",
        }
        for field in allowed_fields:
            if field in config_values and config_values[field] is not None:
                value = config_values[field]
                if field == "use_device_map":
                    kwargs[field] = bool(value)
                else:
                    kwargs[field] = value

    if weights_path and weights_path.is_dir() and "cache_dir" not in kwargs:
        kwargs["cache_dir"] = str(weights_path)

    pose_cfg = PoseLoadConfig(**kwargs)
    return pose_cfg, float(keypoint_threshold)
