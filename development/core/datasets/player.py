"""Player detection dataset built on :class:`BaseSequenceDataset`."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from .base_sequence import BaseSequenceDataset
from ..data_core import coco_io, replay, targets as target_utils


class PlayerSequenceDataset(BaseSequenceDataset):
    """Sequential detection dataset returning clip-aligned bboxes/classes."""

    def __init__(
        self,
        *,
        annotation_file: Optional[str | bytes] = None,
        image_dir: Optional[str] = None,
        coco: Optional[Mapping[str, Any]] = None,
        sequence_length: int,
        frame_stride: int,
        target_category: str = "player",
        min_box_size: float = 1.0,
        image_size: Optional[Sequence[int]] = None,
        drop_short_clips: bool = False,
        allow_partial_last: bool = False,
        transform: Optional[Any] = None,
        normalize_mean: Sequence[float] = (0.485, 0.456, 0.406),
        normalize_std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.target_category = target_category
        self.min_box_size = float(min_box_size)
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

        resolved = coco_io.resolve_category_id(self.coco, self.target_category)
        if resolved is None:
            raise ValueError(f"Category '{self.target_category}' not found in annotations.")
        self.target_category_id = int(resolved)

    # ------------------------------------------------------------------
    def _build_default_transform(self):
        try:
            import albumentations as A
        except Exception:  # pragma: no cover - Albumentations not installed
            return lambda sample: sample

        ops = []
        if self.image_size is not None:
            ops.append(A.Resize(self.image_size[0], self.image_size[1]))
        ops.append(A.Normalize(mean=list(self.normalize_mean), std=list(self.normalize_std)))
        pipeline = A.ReplayCompose(
            ops,
            bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]),
        )
        return replay.make_clip_replay_adapter(
            pipeline,
            keypoints_field="keypoints",
            bboxes_field="bboxes",
            classes_field="classes",
        )

    def _frame_targets(self, frame, annotations):
        bboxes, classes = target_utils.extract_player_bboxes_classes(
            annotations,
            category_id=self.target_category_id,
            min_box_size=self.min_box_size,
        )
        return {
            "bboxes": bboxes,
            "classes": classes,
        }

    def _finalize_sample(self, sample, *, payloads, metadata):
        targets = dict(sample.get("targets", {}))
        targets["bboxes"] = [self._ensure_bbox_list(frame) for frame in self._ensure_sequence(targets.get("bboxes"))]
        targets["classes"] = [self._ensure_class_list(frame) for frame in self._ensure_sequence(targets.get("classes"))]
        return {
            "inputs": sample["inputs"],
            "targets": targets,
            "metadata": sample.get("metadata", metadata),
        }

    # ------------------------------------------------------------------
    def _ensure_sequence(self, seq):
        seq = [] if seq is None else list(seq)
        if len(seq) < self.sequence_length:
            seq.extend([] for _ in range(self.sequence_length - len(seq)))
        return seq[: self.sequence_length]

    @staticmethod
    def _ensure_bbox_list(frame_entry):
        return [list(map(float, bbox)) for bbox in (frame_entry or [])]

    @staticmethod
    def _ensure_class_list(frame_entry):
        return [int(lbl) for lbl in (frame_entry or [])]


__all__ = ["PlayerSequenceDataset"]
