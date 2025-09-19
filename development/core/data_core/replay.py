"""Albumentations replay helpers for clip-level augmentation."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

try:  # Optional dependency – enforced when the helper is used.
    import albumentations as A
except Exception:  # pragma: no cover - Albumentations not installed
    A = None  # type: ignore


class AlbumentationsReplayUnavailable(RuntimeError):
    """Raised when Albumentations ReplayCompose helpers are requested without support."""


def _ensure_replay_support(pipeline: Any) -> None:
    if A is None:
        raise AlbumentationsReplayUnavailable("Albumentations must be installed to use clip replay helpers.")
    if isinstance(pipeline, A.ReplayCompose):
        return
    # Compose is acceptable – replay just runs per-frame without shared params.


def make_clip_replay_adapter(
    pipeline: Any,
    *,
    keypoints_field: str = "keypoints",
    bboxes_field: str = "bboxes",
    classes_field: str = "classes",
) -> Any:
    """Adapt an Albumentations pipeline to operate on clip samples.

    The returned callable expects a mapping with keys:
        ``inputs``: torch.Tensor [T, C, H, W]
        ``targets``: dict containing optional ``keypoints``/``bboxes``/``classes``
        ``metadata``: optional context (preserved)

    Any remaining keys are forwarded untouched. The adapter guarantees that the
    same augmentation parameters are applied to every frame when ``pipeline`` is
    an Albumentations ``ReplayCompose``; otherwise each frame is processed
    independently (matching ``Compose`` semantics).
    """

    _ensure_replay_support(pipeline)

    def to_hwc_uint8(x: torch.Tensor) -> np.ndarray:
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("'inputs' must be a torch.Tensor with shape [T,C,H,W].")
        tensor = x.detach()
        if tensor.dtype.is_floating_point:
            tensor = (tensor.clamp(0, 1) * 255).to(torch.uint8)
        return tensor.permute(0, 2, 3, 1).cpu().numpy()

    def image_to_tensor(image: Any) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            tensor = image.detach().clone()
            orig_dtype = tensor.dtype
            if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
                tensor = tensor
            elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(2, 0, 1)
            else:
                raise ValueError("Transformed tensor image must be 3D (C,H,W) or (H,W,C).")
            tensor = tensor.to(torch.float32)
            if orig_dtype == torch.uint8:
                tensor = tensor / 255.0
            return tensor

        arr = np.asarray(image)
        if arr.ndim != 3:
            raise ValueError("Transformed image must be rank-3 array.")
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            tensor = torch.from_numpy(arr)
        else:
            tensor = torch.from_numpy(np.transpose(arr, (2, 0, 1)))
        orig_dtype = tensor.dtype
        tensor = tensor.to(torch.float32)
        if orig_dtype == torch.uint8:
            tensor = tensor / 255.0
        return tensor

    def normalise_sequence(data: Any) -> Optional[List[Any]]:
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            return [data[i].detach().cpu().tolist() for i in range(data.shape[0])]
        if isinstance(data, np.ndarray):
            return [np.array(data[i]).tolist() for i in range(data.shape[0])]
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
            return [list(item) if isinstance(item, (list, tuple)) else item for item in data]
        raise TypeError("Sequence data must be list/tuple/ndarray/Tensor.")

    def convert_bboxes(bboxes: Optional[List[Any]]) -> Optional[List[List[tuple[float, float, float, float]]]]:
        if bboxes is None:
            return None
        converted: List[List[tuple[float, float, float, float]]] = []
        for frame in bboxes:
            converted.append([tuple(map(float, bbox)) for bbox in frame])
        return converted

    def convert_keypoints(keypoints: Optional[List[Any]]) -> Optional[List[List[tuple[float, float]]]]:
        if keypoints is None:
            return None
        converted: List[List[tuple[float, float]]] = []
        for frame in keypoints:
            converted.append([tuple(map(float, kp)) for kp in frame])
        return converted

    def convert_classes(classes: Optional[List[Any]]) -> Optional[List[List[int]]]:
        if classes is None:
            return None
        converted: List[List[int]] = []
        for frame in classes:
            converted.append([int(lbl) for lbl in frame])
        return converted

    def _apply(sample: Mapping[str, Any]) -> Dict[str, Any]:
        if "inputs" not in sample:
            raise KeyError("Sample must include 'inputs'.")
        images_uint8 = to_hwc_uint8(sample["inputs"])
        targets_in = dict(sample.get("targets", {}))

        keypoints_seq = convert_keypoints(normalise_sequence(targets_in.get(keypoints_field)))
        bboxes_seq = convert_bboxes(normalise_sequence(targets_in.get(bboxes_field)))
        classes_seq = convert_classes(normalise_sequence(targets_in.get(classes_field)))

        def frame_kwargs(frame_index: int) -> Dict[str, Any]:
            kwargs: Dict[str, Any] = {"image": images_uint8[frame_index]}
            if keypoints_seq is not None:
                kwargs["keypoints"] = keypoints_seq[frame_index]
            if bboxes_seq is not None:
                kwargs["bboxes"] = bboxes_seq[frame_index]
                if classes_seq is not None:
                    kwargs["class_labels"] = classes_seq[frame_index]
            return kwargs

        first = pipeline(**frame_kwargs(0))
        replay = first.get("replay") if A is not None and isinstance(pipeline, A.ReplayCompose) else None

        images_out = [image_to_tensor(first["image"])]
        keypoints_out = [first.get("keypoints", [])] if keypoints_seq is not None else None
        bboxes_out = [first.get("bboxes", [])] if bboxes_seq is not None else None
        classes_out = [first.get("class_labels", [])] if classes_seq is not None else None

        for index in range(1, images_uint8.shape[0]):
            out = (
                A.ReplayCompose.replay(replay, **frame_kwargs(index))
                if replay is not None
                else pipeline(**frame_kwargs(index))
            )
            images_out.append(image_to_tensor(out["image"]))
            if keypoints_out is not None:
                keypoints_out.append(out.get("keypoints", []))
            if bboxes_out is not None:
                bboxes_out.append(out.get("bboxes", []))
            if classes_out is not None:
                classes_out.append(out.get("class_labels", []))

        images_tensor = torch.stack(images_out, dim=0).contiguous()
        targets_out = dict(targets_in)
        if keypoints_out is not None:
            targets_out[keypoints_field] = keypoints_out
        if bboxes_out is not None:
            targets_out[bboxes_field] = bboxes_out
        if classes_out is not None:
            targets_out[classes_field] = classes_out

        out_sample: Dict[str, Any] = dict(sample)
        out_sample["inputs"] = images_tensor
        out_sample["targets"] = targets_out
        return out_sample

    return _apply


__all__ = ["AlbumentationsReplayUnavailable", "make_clip_replay_adapter"]
