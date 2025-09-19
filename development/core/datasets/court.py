"""Court keypoint dataset built on :class:`BaseSequenceDataset`."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch

from .base_sequence import BaseSequenceDataset
from ..data_core import replay, targets as target_utils


class CourtKeypointDataset(BaseSequenceDataset):
    """Single-frame keypoint dataset (sequence length = 1)."""

    def __init__(
        self,
        *,
        annotation_file: Optional[str | bytes] = None,
        image_dir: Optional[str] = None,
        coco: Optional[Mapping[str, Any]] = None,
        heatmap_size: Sequence[int],
        heatmap_sigma: float = 2.0,
        image_size: Optional[Sequence[int]] = None,
        sequence_length: int = 1,
        frame_stride: int = 1,
        drop_short_clips: bool = False,
        allow_partial_last: bool = False,
        transform: Optional[Any] = None,
        normalize_mean: Sequence[float] = (0.485, 0.456, 0.406),
        normalize_std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        if sequence_length != 1:
            raise ValueError("CourtKeypointDataset currently supports sequence_length=1.")
        self.heatmap_size = tuple(int(v) for v in heatmap_size)
        self.heatmap_sigma = float(heatmap_sigma)
        self.image_size = tuple(int(v) for v in image_size) if image_size is not None else None
        self.normalize_mean = tuple(float(v) for v in normalize_mean)
        self.normalize_std = tuple(float(v) for v in normalize_std)

        super().__init__(
            annotation_file=annotation_file,
            image_dir=image_dir,
            coco=coco,
            sequence_length=sequence_length,
            frame_stride=frame_stride,
            drop_short_clips=drop_short_clips,
            allow_partial_last=allow_partial_last,
            transform=transform,
        )

    def _build_default_transform(self):
        try:
            import albumentations as A
        except Exception:  # pragma: no cover - Albumentations not installed
            return lambda sample: sample

        ops = []
        if self.image_size is not None:
            ops.append(A.Resize(self.image_size[0], self.image_size[1]))
        ops.append(A.Normalize(mean=list(self.normalize_mean), std=list(self.normalize_std)))
        pipeline = A.Compose(
            ops,
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )
        return replay.make_clip_replay_adapter(pipeline, keypoints_field="keypoints")

    def _frame_targets(self, frame, annotations):
        keypoints = []
        visibility = []
        for ann in annotations:
            coords = ann.get("keypoints")
            if not isinstance(coords, Sequence):
                continue
            iter_points = zip(*[iter(coords)] * 3)
            keypoints = []
            visibility = []
            for x, y, v in iter_points:
                keypoints.append((float(x), float(y)))
                visibility.append(int(v) > 0)
            break  # assume single annotation per image
        return {
            "keypoints": keypoints,
            "visibility": visibility,
        }

    def _finalize_sample(self, sample, *, payloads, metadata):
        inputs = sample["inputs"]
        targets = dict(sample.get("targets", {}))
        keypoints_seq = self._ensure_sequence(targets.get("keypoints"), fill_value=[])
        visibility_seq = self._ensure_sequence(targets.get("visibility"), fill_value=[])

        image_hw = (inputs.shape[-2], inputs.shape[-1])
        scaled = [
            target_utils.scale_points(frame_points, source_size=image_hw, target_size=self.heatmap_size)
            for frame_points in keypoints_seq
        ]
        heatmaps = target_utils.make_heatmaps_xy(
            scaled,
            size_hw=self.heatmap_size,
            sigma=self.heatmap_sigma,
            visibility=visibility_seq,
        )
        heatmaps_tensor = torch.from_numpy(heatmaps[0]).float()  # [K,H,W]

        targets["keypoints"] = keypoints_seq[0]
        targets["visibility"] = visibility_seq[0]
        targets["heatmaps"] = heatmaps_tensor

        return {
            "inputs": inputs,
            "targets": targets,
            "metadata": sample.get("metadata", metadata),
        }

    def _ensure_sequence(self, seq, *, fill_value=None):
        seq = [] if seq is None else list(seq)
        if len(seq) < self.sequence_length:
            seq.extend([self._clone_fill(fill_value) for _ in range(self.sequence_length - len(seq))])
        return seq[: self.sequence_length]

    @staticmethod
    def _clone_fill(value):
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple):
            return list(value)
        return value


__all__ = ["CourtKeypointDataset"]
