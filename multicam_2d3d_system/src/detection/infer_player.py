"""Player detection + pose inference wrapper."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import cv2  # type: ignore
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from trained_models.player_analysis.dino_detr_pose import (
    DinoDetrPoseLoadConfig,
    load_dino_detr_pose,
    rescale_keypoints_to_original,
)
from trained_models.player_analysis.rt_detr.rtdetr_loader import (
    RTDetrLoadConfig,
    load_hf_rtdetr_with_ckpt,
)
from trained_models.player_analysis.vit_pose.vit_pose_loader import (
    PoseLoadConfig,
    load_pose_from_hub,
)

from ..dataio import formats

_LOGGER = logging.getLogger(__name__)


def run_player_inference(
    video_paths: Iterable[Path],
    cfg: DictConfig,
) -> list[formats.FrameDetections2D]:
    """Run the configured player detector/pose pipeline and return detection records."""

    pose_mode = str(cfg.get("pose_mode", "two_stage")).lower()
    _LOGGER.info("Player detection pose_mode='%s'", pose_mode)
    if pose_mode == "two_stage":
        return _run_two_stage_pipeline(video_paths, cfg.get("two_stage"))
    if pose_mode == "single_stage":
        return _run_single_stage_pipeline(video_paths, cfg.get("single_stage"))

    raise ValueError(f"Unsupported detection.player.pose_mode='{pose_mode}'. Expected 'two_stage' or 'single_stage'.")


def _run_two_stage_pipeline(
    video_paths: Iterable[Path],
    cfg: DictConfig | None,
) -> list[formats.FrameDetections2D]:
    if cfg is None:
        raise ValueError("two_stage configuration is required when pose_mode='two_stage'")

    cfg_dict = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg
    detector_cfg_path = Path(cfg_dict.get("detector_config"))
    pose_cfg_path = Path(cfg_dict.get("pose_config"))
    detector_overrides = cfg_dict.get("detector_overrides", {})
    pose_overrides = cfg_dict.get("pose_overrides", {})
    score_threshold = float(cfg_dict.get("score_threshold", 0.5))
    max_players = int(cfg_dict.get("max_players", 8))

    det_cfg = _load_dataclass_with_overrides(RTDetrLoadConfig, detector_cfg_path, detector_overrides)
    pose_cfg = _load_dataclass_with_overrides(PoseLoadConfig, pose_cfg_path, pose_overrides)

    detector_model, detector_processor, det_device = load_hf_rtdetr_with_ckpt(det_cfg)
    pose_model, pose_processor, pose_device = load_pose_from_hub(pose_cfg)

    detector_model.eval()
    pose_model.eval()

    results: list[formats.FrameDetections2D] = []

    for video_path in video_paths:
        for frame_idx, timestamp, frame_bgr in _iter_video_frames(video_path):
            image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            height, width = frame_bgr.shape[:2]

            det_inputs = detector_processor(images=pil_image, return_tensors="pt")
            det_inputs = {k: v.to(det_device) if torch.is_tensor(v) else v for k, v in det_inputs.items()}

            with torch.inference_mode():
                det_outputs = detector_model(**det_inputs)

            target_sizes = torch.tensor([[height, width]], device=det_device)
            det_processed = detector_processor.post_process_object_detection(
                det_outputs,
                threshold=score_threshold,
                target_sizes=target_sizes,
            )[0]

            boxes = det_processed["boxes"]
            labels = det_processed["labels"].to(torch.int64)
            scores = det_processed["scores"]

            keep_mask = labels == 0  # 0 = person for COCO-trained RT-DETR
            if keep_mask.any():
                boxes = boxes[keep_mask]
                scores = scores[keep_mask]
            else:
                boxes = torch.zeros((0, 4), device=boxes.device)
                scores = torch.zeros((0,), device=scores.device)

            if boxes.size(0) > max_players:
                top_scores, indices = torch.topk(scores, max_players)
                boxes = boxes[indices]
                scores = top_scores

            detections = []
            keypoints_result: Sequence[Sequence[Sequence[float]]] = []

            if boxes.numel() > 0:
                boxes_xyxy = boxes.cpu()
                boxes_xywh = torch.stack(
                    [
                        boxes_xyxy[:, 0],
                        boxes_xyxy[:, 1],
                        boxes_xyxy[:, 2] - boxes_xyxy[:, 0],
                        boxes_xyxy[:, 3] - boxes_xyxy[:, 1],
                    ],
                    dim=1,
                )
                boxes_np = boxes_xywh.numpy()

                pose_inputs = pose_processor(
                    pil_image,
                    boxes=[boxes_np],
                    return_tensors="pt",
                )
                pose_inputs = {k: v.to(pose_device) if torch.is_tensor(v) else v for k, v in pose_inputs.items()}

                with torch.inference_mode():
                    pose_outputs = pose_model(**pose_inputs)

                pose_results = pose_processor.post_process_pose_estimation(
                    pose_outputs,
                    boxes=[boxes_np],
                )
                keypoints_result = pose_results[0] if pose_results else []

                for kp_array, score, box_xyxy in zip(keypoints_result, scores.cpu(), boxes_xyxy, strict=False):
                    kp_list = [(float(x), float(y), float(v)) for x, y, v in kp_array]
                    bbox = _bbox_from_keypoints(kp_list, fallback=box_xyxy.tolist())
                    detections.append({
                        "cls": "player",
                        "score": float(score),
                        "bbox": bbox,
                        "keypoints": kp_list,
                    })
            frame_record: formats.FrameDetections2D = {
                "camera_id": video_path.stem,
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "detections": detections,
                "image_size": (height, width),
            }
            results.append(frame_record)

    return results


def _run_single_stage_pipeline(
    video_paths: Iterable[Path],
    cfg: DictConfig | None,
) -> list[formats.FrameDetections2D]:
    if cfg is None:
        raise ValueError("single_stage configuration is required when pose_mode='single_stage'")

    cfg_dict = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg
    pose_cfg_path = Path(cfg_dict.get("pose_config"))
    pose_overrides = cfg_dict.get("pose_overrides", {})
    score_threshold = float(cfg_dict.get("score_threshold", 0.35))
    max_players = int(cfg_dict.get("max_players", 8))
    min_visibility = float(cfg_dict.get("min_visibility", 0.0))

    pose_cfg = _load_dataclass_with_overrides(DinoDetrPoseLoadConfig, pose_cfg_path, pose_overrides)

    model, preprocess, postprocessor, device = load_dino_detr_pose(pose_cfg)
    model.eval()

    results: list[formats.FrameDetections2D] = []

    for video_path in video_paths:
        for frame_idx, timestamp, frame_bgr in _iter_video_frames(video_path):
            image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            processed = preprocess(pil_image)
            tensor = processed.unsqueeze(0).to(device)
            height, width = frame_bgr.shape[:2]

            with torch.inference_mode():
                outputs = model(tensor)

            proc_hw = torch.tensor([[tensor.shape[-2], tensor.shape[-1]]], device=device)
            detections = postprocessor(outputs, proc_hw)
            detections = rescale_keypoints_to_original(
                detections,
                processed_hw=proc_hw[0],
                original_hw=(height, width),
            )

            filtered = _filter_and_format_single_stage(
                detections,
                score_threshold=score_threshold,
                max_players=max_players,
                min_visibility=min_visibility,
            )

            frame_record: formats.FrameDetections2D = {
                "camera_id": video_path.stem,
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "detections": filtered,
                "image_size": (height, width),
            }
            results.append(frame_record)

    return results


def _filter_and_format_single_stage(
    detections: Sequence[dict[str, Any]],
    *,
    score_threshold: float,
    max_players: int,
    min_visibility: float,
) -> list[formats.Detection2DEntry]:
    entries: list[formats.Detection2DEntry] = []

    for det in detections:
        scores = torch.as_tensor(det.get("scores", []), dtype=torch.float32).flatten()
        if scores.numel() == 0:
            continue

        labels = torch.as_tensor(det.get("labels", []), dtype=torch.int64).flatten()
        keypoints = torch.as_tensor(det.get("keypoints", []), dtype=torch.float32)
        if keypoints.ndim == 1:
            # Expect [num_queries * num_dims]
            if scores.numel() == 0:
                continue
            num_dims = max(1, keypoints.numel() // scores.numel())
            keypoints = keypoints.view(scores.numel(), num_dims)
        if keypoints.ndim == 2:
            num_points = keypoints.shape[1] // 3
            keypoints = keypoints.view(scores.numel(), num_points, 3)

        order = torch.argsort(scores, descending=True)

        selected = 0
        for idx in order:
            if selected >= max_players:
                break
            score = float(scores[idx])
            if score < score_threshold:
                continue
            if labels.numel() > idx and int(labels[idx]) not in (0,):
                # Only keep player class (0)
                continue

            kp_tensor = keypoints[idx]
            if kp_tensor.numel() == 0:
                continue

            if min_visibility > 0:
                vis_mask = kp_tensor[:, 2] >= min_visibility
                if not vis_mask.any():
                    continue
                xs = kp_tensor[vis_mask, 0]
                ys = kp_tensor[vis_mask, 1]
            else:
                xs = kp_tensor[:, 0]
                ys = kp_tensor[:, 1]

            bbox = _bbox_from_xy(xs, ys)
            kp_list = [(float(x), float(y), float(v)) for x, y, v in kp_tensor.tolist()]

            entry: formats.Detection2DEntry = {
                "cls": "player",
                "score": score,
                "bbox": bbox,
                "keypoints": kp_list,
            }
            entries.append(entry)
            selected += 1

    return entries


def _bbox_from_keypoints(
    keypoints: Sequence[tuple[float, float, float]],
    *,
    fallback: Sequence[float] | None = None,
) -> formats.BBox:
    xs = [p[0] for p in keypoints if p[2] > 0]
    ys = [p[1] for p in keypoints if p[2] > 0]
    if xs and ys:
        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)
    elif fallback is not None and len(fallback) == 4:
        x_min, y_min, x_max, y_max = fallback
    else:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min))


def _bbox_from_xy(xs: torch.Tensor, ys: torch.Tensor) -> formats.BBox:
    if xs.numel() == 0 or ys.numel() == 0:
        return (0.0, 0.0, 0.0, 0.0)
    x_min = float(xs.min())
    y_min = float(ys.min())
    x_max = float(xs.max())
    y_max = float(ys.max())
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def _iter_video_frames(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Failed to open video {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_idx = 0
    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break
            timestamp = frame_idx / fps if fps > 0 else float(frame_idx)
            yield frame_idx, timestamp, frame
            frame_idx += 1
    finally:
        capture.release()


def _load_dataclass_with_overrides(cls, config_path: Path, overrides: dict[str, Any]) -> Any:
    cfg = cls.from_yaml(str(config_path))
    for key, value in (overrides or {}).items():
        if not hasattr(cfg, key):
            raise AttributeError(f"Unknown override '{key}' for config {cls.__name__}")
        setattr(cfg, key, value)
    return cfg
