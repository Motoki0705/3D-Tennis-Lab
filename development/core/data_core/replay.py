"""Albumentations replay helpers for clip-level augmentation (pickle-safe)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

try:  # Optional dependency – enforced when the helper is used.
    import albumentations as A
except Exception:  # pragma: no cover - Albumentations not installed
    A = None  # type: ignore


class AlbumentationsReplayUnavailable(RuntimeError):
    """Raised when Albumentations ReplayCompose helpers are requested without support."""


# ------------------------------ utils (top-level; pickle-safe) ------------------------------


def _ensure_replay_support(pipeline: Any) -> None:
    if A is None:
        raise AlbumentationsReplayUnavailable("Albumentations must be installed to use clip replay helpers.")
    # ReplayCompose も Compose も受け付ける（Compose はフレームごとに独立適用）


def _to_hwc_uint8_sequence(x: torch.Tensor) -> np.ndarray:
    """[T,C,H,W] float[0..1] or uint8 -> [T,H,W,C] uint8 (CPU numpy)."""
    if not isinstance(x, torch.Tensor) or x.ndim != 4:
        raise ValueError("'inputs' must be a torch.Tensor with shape [T,C,H,W].")
    t = x.detach()
    if t.dtype.is_floating_point:
        t = (t.clamp(0, 1) * 255).to(torch.uint8)
    return t.permute(0, 2, 3, 1).cpu().numpy()


def _image_to_tensor(image: Any) -> torch.Tensor:
    """(H,W,C) or (C,H,W) ndarray/torch.Tensor/uint8 -> float32(C,H,W) in [0,1]."""
    if isinstance(image, torch.Tensor):
        tensor = image.detach().clone()
        if tensor.ndim != 3:
            raise ValueError("Transformed tensor image must be 3D (C,H,W) or (H,W,C).")
        if tensor.shape[0] not in (1, 3) and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        elif tensor.shape[0] not in (1, 3) and tensor.shape[-1] not in (1, 3):
            raise ValueError("Transformed tensor image must be 3D with channel dim size 1 or 3.")
        orig_dtype = tensor.dtype
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


def _normalise_sequence(data: Any) -> Optional[List[Any]]:
    """Accept torch.Tensor/np.ndarray/Sequence -> list-of-items (or None)."""
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return [data[i].detach().cpu().tolist() for i in range(data.shape[0])]
    if isinstance(data, np.ndarray):
        return [np.array(data[i]).tolist() for i in range(data.shape[0])]
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return [list(item) if isinstance(item, (list, tuple)) else item for item in data]
    raise TypeError("Sequence data must be list/tuple/ndarray/Tensor.")


def _convert_bboxes(bboxes: Optional[List[Any]]) -> Optional[List[List[Tuple[float, float, float, float]]]]:
    if bboxes is None:
        return None
    converted: List[List[Tuple[float, float, float, float]]] = []
    for frame in bboxes:
        converted.append([tuple(map(float, bbox)) for bbox in frame])
    return converted


def _convert_keypoints(keypoints: Optional[List[Any]]) -> Optional[List[List[Tuple[float, float]]]]:
    if keypoints is None:
        return None
    converted: List[List[Tuple[float, float]]] = []
    for frame in keypoints:
        converted.append([tuple(map(float, kp)) for kp in frame])
    return converted


def _convert_classes(classes: Optional[List[Any]]) -> Optional[List[List[int]]]:
    if classes is None:
        return None
    converted: List[List[int]] = []
    for frame in classes:
        converted.append([int(lbl) for lbl in frame])
    return converted


# ------------------------------ adapter (top-level callable) ------------------------------


class _ClipReplayAdapter:
    """Pickle-safe callable that applies Albumentations to a clip sample.

    入力 sample 形式:
      inputs : torch.Tensor [T,C,H,W]
      targets: dict (任意。keypoints/bboxes/classes を含むかも)
      metadata: 任意

    出力:
      inputs : torch.Tensor [T,C,H,W]  (変換後)
      targets: dict (keypoints/bboxes/classes がフレームごとに変換後)
    """

    def __init__(
        self,
        pipeline: Any,
        *,
        keypoints_field: str = "keypoints",
        bboxes_field: str = "bboxes",
        classes_field: str = "classes",
    ) -> None:
        _ensure_replay_support(pipeline)
        self.pipeline = pipeline
        self.keypoints_field = keypoints_field
        self.bboxes_field = bboxes_field
        self.classes_field = classes_field

    def _frame_kwargs(
        self,
        images_uint8: np.ndarray,
        idx: int,
        keypoints_seq: Optional[List[List[Tuple[float, float]]]],
        bboxes_seq: Optional[List[List[Tuple[float, float, float, float]]]],
        classes_seq: Optional[List[List[int]]],
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"image": images_uint8[idx]}
        if keypoints_seq is not None:
            kwargs["keypoints"] = keypoints_seq[idx]
        if bboxes_seq is not None:
            kwargs["bboxes"] = bboxes_seq[idx]
            if classes_seq is not None:
                kwargs["class_labels"] = classes_seq[idx]
        return kwargs

    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        if "inputs" not in sample:
            raise KeyError("Sample must include 'inputs'.")

        images_uint8 = _to_hwc_uint8_sequence(sample["inputs"])
        targets_in = dict(sample.get("targets", {}))

        keypoints_seq = _convert_keypoints(_normalise_sequence(targets_in.get(self.keypoints_field)))
        bboxes_seq = _convert_bboxes(_normalise_sequence(targets_in.get(self.bboxes_field)))
        classes_seq = _convert_classes(_normalise_sequence(targets_in.get(self.classes_field)))

        # 最初のフレームを適用して（必要なら）replay を取得
        first_kwargs = self._frame_kwargs(images_uint8, 0, keypoints_seq, bboxes_seq, classes_seq)
        first_out = self.pipeline(**first_kwargs)
        replay = first_out.get("replay") if (A is not None and isinstance(self.pipeline, A.ReplayCompose)) else None

        images_out = [_image_to_tensor(first_out["image"])]
        keypoints_out = [first_out.get("keypoints", [])] if keypoints_seq is not None else None
        bboxes_out = [first_out.get("bboxes", [])] if bboxes_seq is not None else None
        classes_out = [first_out.get("class_labels", [])] if classes_seq is not None else None

        for index in range(1, images_uint8.shape[0]):
            kwargs = self._frame_kwargs(images_uint8, index, keypoints_seq, bboxes_seq, classes_seq)
            out = A.ReplayCompose.replay(replay, **kwargs) if replay is not None else self.pipeline(**kwargs)

            images_out.append(_image_to_tensor(out["image"]))
            if keypoints_out is not None:
                keypoints_out.append(out.get("keypoints", []))
            if bboxes_out is not None:
                bboxes_out.append(out.get("bboxes", []))
            if classes_out is not None:
                classes_out.append(out.get("class_labels", []))

        images_tensor = torch.stack(images_out, dim=0).contiguous()
        targets_out = dict(targets_in)
        if keypoints_out is not None:
            targets_out[self.keypoints_field] = keypoints_out
        if bboxes_out is not None:
            targets_out[self.bboxes_field] = bboxes_out
        if classes_out is not None:
            targets_out[self.classes_field] = classes_out

        out_sample: Dict[str, Any] = dict(sample)
        out_sample["inputs"] = images_tensor
        out_sample["targets"] = targets_out
        return out_sample


# ------------------------------ public factory ------------------------------


def make_clip_replay_adapter(
    pipeline: Any,
    *,
    keypoints_field: str = "keypoints",
    bboxes_field: str = "bboxes",
    classes_field: str = "classes",
) -> Any:
    """Adapt an Albumentations pipeline to operate on clip samples (pickle-safe)."""
    return _ClipReplayAdapter(
        pipeline,
        keypoints_field=keypoints_field,
        bboxes_field=bboxes_field,
        classes_field=classes_field,
    )


__all__ = ["AlbumentationsReplayUnavailable", "make_clip_replay_adapter"]
