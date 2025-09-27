"""Collate utilities for single-frame pose samples."""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def collate_pose_seq_t1(batch: List[Dict[str, Any]]):
    """Flatten [T=1] sequences into images + DETRPose targets list."""

    images: List[torch.Tensor] = []
    targets: List[Dict[str, torch.Tensor]] = []

    for sample in batch:
        inputs = sample["inputs"]
        assert inputs.ndim == 4 and inputs.shape[0] == 1, "Expected [T=1,C,H,W] inputs"
        image = inputs[0]
        images.append(image)

        frame_targets = sample.get("targets", {})
        boxes_seq = frame_targets.get("boxes", [])
        labels_seq = frame_targets.get("labels", [])
        keypoints_seq = frame_targets.get("keypoints", [])
        area_seq = frame_targets.get("area", [])
        crowd_seq = frame_targets.get("iscrowd", [])
        orig_size_seq = frame_targets.get("orig_size", [])
        size_seq = frame_targets.get("size", [])

        keypoint_dim = keypoints_seq[0].shape[-1] if keypoints_seq else 51

        boxes = boxes_seq[0] if boxes_seq else torch.zeros((0, 4), dtype=torch.float32)
        labels = labels_seq[0] if labels_seq else torch.zeros((0,), dtype=torch.int64)
        keypoints = keypoints_seq[0] if keypoints_seq else torch.zeros((0, keypoint_dim), dtype=torch.float32)
        area = area_seq[0] if area_seq else torch.zeros((0,), dtype=torch.float32)
        crowd = crowd_seq[0] if crowd_seq else torch.zeros((0,), dtype=torch.int64)
        orig_size = (
            orig_size_seq[0] if orig_size_seq else torch.tensor([image.size(2), image.size(1)], dtype=torch.int64)
        )
        size = size_seq[0] if size_seq else torch.tensor([image.size(1), image.size(2)], dtype=torch.int64)

        targets.append({
            "boxes": boxes.to(dtype=torch.float32),
            "labels": labels.to(dtype=torch.int64),
            "keypoints": keypoints.to(dtype=torch.float32),
            "area": area.to(dtype=torch.float32),
            "iscrowd": crowd.to(dtype=torch.int64),
            "orig_size": orig_size.to(dtype=torch.int64),
            "size": size.to(dtype=torch.int64),
        })

    images_tensor = torch.stack(images, dim=0).contiguous()
    return images_tensor, targets


__all__ = ["collate_pose_seq_t1"]
