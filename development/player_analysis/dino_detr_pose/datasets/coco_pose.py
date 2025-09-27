"""COCO-style dataset returning person pose annotations for MPPE training."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from development.core.datasets.base_sequence import BaseSequenceDataset
from development.core.data_core import coco_io


KeypointList = List[Tuple[float, float, float]]
BBox = Tuple[float, float, float, float]


def _xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [N,4] boxes from xyxy to cxcywh."""
    if boxes.numel() == 0:
        return boxes.view(-1, 4)
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


class CocoPlayerPoseDataset(BaseSequenceDataset):
    """MPPE dataset that normalises COCO 17-keypoint annotations for DETRPose."""

    def __init__(
        self,
        *,
        annotation_file: Optional[str | bytes] = None,
        image_dir: Optional[str] = None,
        coco: Optional[Mapping[str, Any]] = None,
        sequence_length: int = 1,
        frame_stride: int = 1,
        image_size: Sequence[int] = (320, 640),  # (H, W)
        normalize_mean: Sequence[float] = (0.485, 0.456, 0.406),
        normalize_std: Sequence[float] = (0.229, 0.224, 0.225),
        target_categories: Sequence[str | int] = ("person",),
        min_keypoints: int = 1,
        min_box_area: float = 1.0,
    ) -> None:
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.normalize_mean = torch.tensor(normalize_mean, dtype=torch.float32).view(1, -1, 1, 1)
        self.normalize_std = torch.tensor(normalize_std, dtype=torch.float32).view(1, -1, 1, 1)
        self.min_keypoints = int(min_keypoints)
        self.min_box_area = float(min_box_area)
        self.num_keypoints = 17  # fixed for COCO MPPE

        coco_obj = coco if coco is not None else coco_io.load_coco(annotation_file)  # type: ignore[arg-type]
        target_ids: List[int] = []
        for category in target_categories:
            resolved = coco_io.resolve_category_id(coco_obj, category)
            if resolved is None:
                raise ValueError(f"Category '{category}' not found in COCO annotations.")
            target_ids.append(int(resolved))
        self.target_category_ids = tuple(sorted(set(target_ids)))

        super().__init__(
            annotation_file=annotation_file,
            image_dir=image_dir,
            coco=coco_obj,
            sequence_length=sequence_length,
            frame_stride=frame_stride,
            drop_short_clips=False,
            allow_partial_last=False,
            transform=lambda sample: sample,
        )

    # ------------------------------------------------------------------
    # Base hooks
    # ------------------------------------------------------------------
    def _build_default_transform(self):  # pragma: no cover - handled via lambda in __init__
        return lambda sample: sample

    def _frame_targets(
        self,
        frame: Mapping[str, Any],
        annotations: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        boxes: List[BBox] = []
        classes: List[int] = []
        keypoints: List[KeypointList] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        for ann in annotations:
            category_id = int(ann.get("category_id", -1))
            if category_id not in self.target_category_ids:
                continue

            kp = ann.get("keypoints")
            if not isinstance(kp, Sequence) or len(kp) < self.num_keypoints * 3:
                continue
            visibility_sum = 0
            kp_triplets: KeypointList = []
            for idx in range(self.num_keypoints):
                x = float(kp[3 * idx])
                y = float(kp[3 * idx + 1])
                v = float(kp[3 * idx + 2])
                visibility_sum += int(v > 0)
                # Keep raw coordinates; zero them out if flagged invisible later.
                kp_triplets.append((x, y, v))
            if visibility_sum < self.min_keypoints:
                continue

            bbox = ann.get("bbox")
            if not isinstance(bbox, Sequence) or len(bbox) != 4:
                continue
            x, y, w, h = map(float, bbox)
            if w <= 0 or h <= 0:
                continue
            if (w * h) < self.min_box_area:
                continue

            boxes.append((x, y, w, h))
            classes.append(0)  # single-class MPPE experiment
            keypoints.append(kp_triplets)
            areas.append(float(ann.get("area", w * h)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        return {
            "bboxes": boxes,
            "classes": classes,
            "keypoints": keypoints,
            "areas": areas,
            "iscrowd": iscrowd,
        }

    def _finalize_sample(
        self,
        sample: Mapping[str, Any],
        *,
        payloads: Sequence[Mapping[str, Any]],
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        inputs = sample["inputs"].float()  # [T,C,H,W]
        T, C, H, W = inputs.shape
        target_h, target_w = self.image_size
        resized = F.interpolate(inputs, size=(target_h, target_w), mode="bilinear", align_corners=False)
        normalised = (resized - self.normalize_mean.to(resized.device)) / self.normalize_std.to(resized.device)

        aggregated = sample.get("targets", {})
        boxes_seq = aggregated.get("bboxes", [])
        classes_seq = aggregated.get("classes", [])
        keypoints_seq = aggregated.get("keypoints", [])
        areas_seq = aggregated.get("areas", [])
        iscrowd_seq = aggregated.get("iscrowd", [])

        boxes_out: List[torch.Tensor] = []
        labels_out: List[torch.Tensor] = []
        keypoints_out: List[torch.Tensor] = []
        area_out: List[torch.Tensor] = []
        crowd_out: List[torch.Tensor] = []
        orig_size_out: List[torch.Tensor] = []
        size_out: List[torch.Tensor] = []

        for frame_idx in range(T):
            payload = payloads[frame_idx]
            orig_h, orig_w = payload.get("image_size", (H, W))
            scale_x = target_w / max(float(orig_w), 1.0)
            scale_y = target_h / max(float(orig_h), 1.0)

            frame_boxes: Sequence[BBox] = boxes_seq[frame_idx] if frame_idx < len(boxes_seq) else []
            frame_labels: Sequence[int] = classes_seq[frame_idx] if frame_idx < len(classes_seq) else []
            frame_keypoints: Sequence[KeypointList] = keypoints_seq[frame_idx] if frame_idx < len(keypoints_seq) else []
            frame_areas: Sequence[float] = areas_seq[frame_idx] if frame_idx < len(areas_seq) else []
            frame_crowd: Sequence[int] = iscrowd_seq[frame_idx] if frame_idx < len(iscrowd_seq) else []

            if frame_boxes:
                xyxy = []
                for x, y, w_box, h_box in frame_boxes:
                    x_scaled = x * scale_x
                    y_scaled = y * scale_y
                    w_scaled = w_box * scale_x
                    h_scaled = h_box * scale_y
                    xyxy.append((x_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled))
                boxes_tensor = torch.tensor(xyxy, dtype=torch.float32)
                boxes_cxcywh = _xyxy_to_cxcywh(boxes_tensor)
                boxes_norm = boxes_cxcywh / torch.tensor([target_w, target_h, target_w, target_h], dtype=torch.float32)
            else:
                boxes_norm = torch.zeros((0, 4), dtype=torch.float32)

            if frame_labels:
                labels_tensor = torch.tensor(frame_labels, dtype=torch.int64)
            else:
                labels_tensor = torch.zeros((0,), dtype=torch.int64)

            if frame_keypoints:
                kp_xy = []
                kp_vis = []
                for person in frame_keypoints:
                    coords: List[float] = []
                    vis: List[float] = []
                    for kx, ky, kv in person:
                        if kv <= 0:
                            coords.extend([0.0, 0.0])
                            vis.append(0.0)
                            continue
                        coords.extend([kx * scale_x, ky * scale_y])
                        vis.append(1.0 if kv >= 2 else float(kv))
                    kp_xy.append(coords)
                    kp_vis.append(vis)
                kp_xy_tensor = torch.tensor(kp_xy, dtype=torch.float32)
                kp_xy_tensor = kp_xy_tensor / torch.tensor(
                    [target_w, target_h] * self.num_keypoints, dtype=torch.float32
                )
                kp_vis_tensor = torch.tensor(kp_vis, dtype=torch.float32)
                keypoints_tensor = torch.cat([kp_xy_tensor, kp_vis_tensor], dim=1)
            else:
                keypoints_tensor = torch.zeros((0, self.num_keypoints * 3), dtype=torch.float32)

            if frame_areas:
                areas_tensor = torch.tensor(frame_areas, dtype=torch.float32) * (scale_x * scale_y)
                areas_tensor = areas_tensor / (target_w * target_h)
            else:
                areas_tensor = torch.zeros((0,), dtype=torch.float32)

            if frame_crowd:
                crowd_tensor = torch.tensor(frame_crowd, dtype=torch.int64)
            else:
                crowd_tensor = torch.zeros((0,), dtype=torch.int64)

            boxes_out.append(boxes_norm)
            labels_out.append(labels_tensor)
            keypoints_out.append(keypoints_tensor)
            area_out.append(areas_tensor)
            crowd_out.append(crowd_tensor)
            orig_size_out.append(torch.tensor([int(orig_w), int(orig_h)], dtype=torch.int64))
            size_out.append(torch.tensor([int(target_h), int(target_w)], dtype=torch.int64))

        targets_final = {
            "boxes": boxes_out,
            "labels": labels_out,
            "keypoints": keypoints_out,
            "area": area_out,
            "iscrowd": crowd_out,
            "orig_size": orig_size_out,
            "size": size_out,
        }

        return {
            "inputs": normalised.contiguous(),
            "targets": targets_final,
            "metadata": metadata,
        }


__all__ = ["CocoPlayerPoseDataset"]
