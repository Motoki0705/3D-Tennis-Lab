"""Shared base class for sequential datasets (ball/player/etc.)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

import torch
from torch.utils.data import Dataset

from ..data_core import coco_io, grouping

FrameRecord = MutableMapping[str, Any]
ClipRecord = Mapping[str, Any]


@dataclass
class SequenceDescriptor:
    clip_idx: int
    frame_indices: List[int]


class BaseSequenceDataset(Dataset):
    """Abstract dataset that handles clip/sequence enumeration and transforms.

    Subclasses implement the lightweight hooks to extract frame-level targets and
    post-process the transformed sample (e.g., build heatmaps, collate bboxes).
    """

    def __init__(
        self,
        *,
        annotation_file: Optional[str | Path] = None,
        image_dir: Optional[str | Path] = None,
        coco: Optional[Mapping[str, Any]] = None,
        sequence_length: int,
        frame_stride: int,
        drop_short_clips: bool = False,
        allow_partial_last: bool = False,
        transform: Optional[Any] = None,
    ) -> None:
        if coco is None:
            if annotation_file is None:
                raise ValueError("Either 'annotation_file' or preloaded 'coco' must be provided.")
            coco = coco_io.load_coco(annotation_file)
        self.coco = coco
        self.annotation_file = annotation_file
        self.image_dir = Path(image_dir) if image_dir is not None else None

        self.sequence_length = int(sequence_length)
        self.frame_stride = int(frame_stride)
        self.drop_short_clips = bool(drop_short_clips)
        self.allow_partial_last = bool(allow_partial_last)

        self.images_map: Dict[int, FrameRecord] = coco_io.index_images(coco)
        self.annotations_by_image = coco_io.collect_annotations_by_image(coco)
        self.clips: List[ClipRecord] = grouping.group_clips(self.images_map.values())
        self._sequences: List[SequenceDescriptor] = self._enumerate_sequences()

        default_transform = self._build_default_transform()
        self.transform = transform if transform is not None else default_transform
        if self.transform is None:
            raise RuntimeError("Dataset requires a transform callable; provide one or enable defaults.")

    # ------------------------------------------------------------------
    # Torch dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        descriptor = self._sequences[index]
        clip = self.clips[descriptor.clip_idx]
        frames = clip.get("frames", [])
        if not frames:
            raise IndexError("Clip has no frames; cannot build sample.")

        frame_records = self._resolve_frame_records(frames, descriptor.frame_indices)
        payloads = [self._build_frame_payload(frame) for frame in frame_records]
        images = [payload["image_tensor"] for payload in payloads]
        inputs = torch.stack(images, dim=0).float()

        raw_targets = self._aggregate_frame_targets(payloads)
        metadata = self._build_metadata(descriptor, payloads)

        sample = {
            "inputs": inputs,
            "targets": raw_targets,
            "metadata": metadata,
        }

        transformed = self.transform(sample)
        if not isinstance(transformed, Mapping):
            raise TypeError("Transform callable must return a mapping.")
        if "inputs" not in transformed:
            raise KeyError("Transform output must include 'inputs'.")
        if "targets" not in transformed:
            transformed = dict(transformed)
            transformed["targets"] = raw_targets

        final_sample = self._finalize_sample(
            transformed,
            payloads=payloads,
            metadata=metadata,
        )

        inputs_tensor = final_sample.get("inputs")
        if not isinstance(inputs_tensor, torch.Tensor) or inputs_tensor.ndim != 4:
            raise ValueError("Final sample must expose 'inputs' as torch.Tensor [T,C,H,W].")
        if inputs_tensor.shape[0] != self.sequence_length:
            raise ValueError(
                f"Dataset must return sequences of length {self.sequence_length}; got {inputs_tensor.shape[0]}."
            )
        return final_sample

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------
    def _build_default_transform(self):  # pragma: no cover - abstract hook
        return None

    def _frame_targets(self, frame: FrameRecord, annotations: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        raise NotImplementedError

    def _finalize_sample(
        self,
        sample: Mapping[str, Any],
        *,
        payloads: Sequence[Mapping[str, Any]],
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _enumerate_sequences(self) -> List[SequenceDescriptor]:
        descriptors: List[SequenceDescriptor] = []
        for clip_idx, clip in enumerate(self.clips):
            frames = clip.get("frames", [])
            total = len(frames)
            if total == 0:
                continue
            sequences = grouping.enumerate_sequences(
                total_frames=total,
                sequence_length=self.sequence_length,
                frame_stride=self.frame_stride,
                allow_partial_last=self.allow_partial_last,
                drop_short_clips=self.drop_short_clips,
            )
            for seq in sequences:
                descriptors.append(SequenceDescriptor(clip_idx=clip_idx, frame_indices=list(seq)))
        return descriptors

    def _resolve_frame_records(
        self,
        frames: Sequence[FrameRecord],
        frame_indices: Sequence[int],
    ) -> List[FrameRecord]:
        resolved: List[FrameRecord] = []
        if not frame_indices:
            raise ValueError("Sequence descriptor is empty; expected at least one frame index.")
        total = len(frames)
        last_valid = frames[min(frame_indices[-1], total - 1)]
        for idx in frame_indices:
            if idx < total:
                resolved.append(frames[idx])
            else:
                resolved.append(last_valid)
        # Pad if sequence shorter than required length.
        while len(resolved) < self.sequence_length:
            resolved.append(resolved[-1])
        return resolved[: self.sequence_length]

    def _build_frame_payload(self, frame: FrameRecord) -> Mapping[str, Any]:
        image_tensor = self._load_image_tensor(frame)
        frame_id = int(frame.get("id"))
        annotations = self.annotations_by_image.get(frame_id, [])
        targets = self._frame_targets(frame, annotations)
        return {
            "frame": frame,
            "image_tensor": image_tensor,
            "targets": targets,
            "frame_id": frame_id,
            "image_path": str(self._resolve_image_path(frame)),
            "image_size": (image_tensor.shape[-2], image_tensor.shape[-1]),
        }

    def _load_image_tensor(self, frame: FrameRecord) -> torch.Tensor:
        import cv2

        path = self._resolve_image_path(frame)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at '{path}'.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return tensor

    def _resolve_image_path(self, frame: Mapping[str, Any]) -> Path:
        candidate = frame.get("original_path") or frame.get("file_name")
        if candidate is None:
            raise ValueError("Frame record must include 'file_name' or 'original_path'.")
        path = Path(candidate)
        if path.is_absolute() or self.image_dir is None:
            return path
        return (self.image_dir / path).resolve()

    def _aggregate_frame_targets(self, payloads: Sequence[Mapping[str, Any]]) -> Dict[str, List[Any]]:
        aggregated: Dict[str, List[Any]] = {}
        for payload in payloads:
            targets = payload.get("targets", {})
            for key, value in targets.items():
                aggregated.setdefault(key, []).append(value)
        return aggregated

    def _build_metadata(
        self,
        descriptor: SequenceDescriptor,
        payloads: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "clip_idx": descriptor.clip_idx,
            "frame_indices": list(descriptor.frame_indices),
            "frame_ids": [payload["frame_id"] for payload in payloads],
            "image_paths": [payload["image_path"] for payload in payloads],
        }


__all__ = ["BaseSequenceDataset", "SequenceDescriptor"]
