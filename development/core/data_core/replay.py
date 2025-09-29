"""Albumentations clip-level replay adapter (COCO bboxes only).

- Always assumes COCO bbox format: (x, y, w, h) in pixels.
- Keypoints are provided/returned as (x, y) in pixels.
- Accepts nested keypoints per frame (e.g., per-instance lists) and flattens.
- Works with A.Compose (applies per frame) and A.ReplayCompose (same params across frames).

Input sample:
    {
      "inputs":  torch.Tensor[T, C, H, W]  # float[0..1] or uint8
      "targets": {
          "bboxes":   list[list[ (x,y,w,h), ... ]],   # COCO pixels
          "classes":  list[list[ int, ... ]],
          "keypoints":list[list[ (x,y) or (x,y,*) , ... ]]  # flat or nested
      }
      "metadata": ...  # passed through unchanged
    }

Output sample:
    Same keys; images transformed to float32 [0..1], and per-frame targets transformed.
"""

from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import albumentations as A
except Exception:  # pragma: no cover
    A = None  # type: ignore


class AlbumentationsReplayUnavailable(RuntimeError):
    """Raised when Albumentations helpers are requested without the package installed."""


# ------------------------------ small utils ------------------------------


def _ensure_albu() -> None:
    if A is None:
        raise AlbumentationsReplayUnavailable("Albumentations must be installed to use this adapter.")


def _tchw_to_hwc_u8(x: torch.Tensor) -> np.ndarray:
    """[T,C,H,W] float[0..1] or uint8 -> [T,H,W,C] uint8 (numpy, CPU)."""
    if not isinstance(x, torch.Tensor) or x.ndim != 4:
        raise ValueError("'inputs' must be a torch.Tensor with shape [T,C,H,W].")
    t = x.detach()
    if t.dtype.is_floating_point:
        t = (t.clamp(0, 1) * 255).to(torch.uint8)
    return t.permute(0, 2, 3, 1).cpu().numpy()


def _to_chw_f32(img: Any) -> torch.Tensor:
    """(H,W,C) or (C,H,W) ndarray/tensor/uint8 -> float32 (C,H,W) in [0,1]."""
    if isinstance(img, torch.Tensor):
        t = img.detach()
        if t.ndim != 3:
            raise ValueError("Transformed image must be rank-3.")
        if t.shape[-1] in (1, 3) and t.shape[0] not in (1, 3):
            t = t.permute(2, 0, 1)
        out = t.to(torch.float32)
        if t.dtype == torch.uint8:
            out = out / 255.0
        return out

    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError("Transformed image must be rank-3 array.")
    if arr.shape[-1] in (1, 3):
        t = torch.from_numpy(arr).permute(2, 0, 1)
    else:
        t = torch.from_numpy(arr)
    out = t.to(torch.float32)
    if out.dtype == torch.uint8:
        out = out / 255.0
    return out


def _as_seq(data: Any) -> Optional[List[Any]]:
    """Accept torch.Tensor / np.ndarray / python Sequence -> list-of-items (or None)."""
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return [data[i].detach().cpu().tolist() for i in range(data.shape[0])]
    if isinstance(data, np.ndarray):
        return [np.array(data[i]).tolist() for i in range(data.shape[0])]
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return [list(x) if isinstance(x, (list, tuple)) else x for x in data]
    raise TypeError("Sequence data must be list/tuple/ndarray/Tensor.")


def _flatten_keypoints_frame(frame: Sequence[Any]) -> List[Tuple[float, float]]:
    """Accept [(x,y[,v]), ...] or [[(x,y[,v]), ...], ...] -> flat [(x,y), ...]."""
    out: List[Tuple[float, float]] = []
    for item in frame:
        if isinstance(item, (list, tuple)) and item and isinstance(item[0], (list, tuple, np.ndarray)):
            # nested group (e.g., per instance)
            for kp in item:
                if len(kp) >= 2:
                    out.append((float(kp[0]), float(kp[1])))
        else:
            if isinstance(item, (list, tuple, np.ndarray)) and len(item) >= 2:
                out.append((float(item[0]), float(item[1])))
    return out


def _convert_bboxes_frame(frame: Sequence[Sequence[float]]) -> List[Tuple[float, float, float, float]]:
    return [tuple(map(float, b)) for b in frame]


def _convert_classes_frame(frame: Sequence[int]) -> List[int]:
    return [int(v) for v in frame]


def _get_bbox_params(pipeline: Any):
    proc = getattr(pipeline, "processors", None)
    if proc is None or not hasattr(proc, "get"):
        return None
    bproc = proc.get("bboxes")
    return getattr(bproc, "params", None)


def _get_keypoint_params(pipeline: Any):
    proc = getattr(pipeline, "processors", None)
    if proc is None or not hasattr(proc, "get"):
        return None
    kproc = proc.get("keypoints")
    return getattr(kproc, "params", None)


def _sanitize_coco_boxes(
    boxes: List[Tuple[float, float, float, float]],
    labels: Optional[List[int]],
    hw: Tuple[int, int],
) -> Tuple[List[Tuple[float, float, float, float]], List[int]]:
    """Clamp COCO (x,y,w,h) to image bounds; always returns a labels list (possibly empty)."""
    H, W = hw
    out_b: List[Tuple[float, float, float, float]] = []
    out_l: List[int] = []
    labels = labels or []
    for i, (x, y, w, h) in enumerate(boxes):
        x = max(0.0, min(float(x), max(0.0, W - 1.0)))
        y = max(0.0, min(float(y), max(0.0, H - 1.0)))
        w = max(0.0, min(float(w), max(0.0, W - x)))
        h = max(0.0, min(float(h), max(0.0, H - y)))
        out_b.append((x, y, w, h))
        if i < len(labels):
            out_l.append(int(labels[i]))
    # If there are more boxes than labels, pad with zero class
    if len(out_l) < len(out_b):
        out_l.extend([0] * (len(out_b) - len(out_l)))
    return out_b, out_l


# ------------------------------ adapter ------------------------------


class ClipReplayAdapter:
    """Apply an Albumentations pipeline to a clip sample (COCO bboxes only)."""

    def __init__(
        self,
        pipeline: Any,
        *,
        keypoints_field: str = "keypoints",
        bboxes_field: str = "bboxes",
        classes_field: str = "classes",
    ) -> None:
        _ensure_albu()
        self.pipeline = pipeline
        self.keypoints_field = keypoints_field
        self.bboxes_field = bboxes_field
        self.classes_field = classes_field

        bp = _get_bbox_params(pipeline)
        kp = _get_keypoint_params(pipeline)

        self._has_bboxes = bp is not None
        self._has_keypoints = kp is not None

        # Whatever label field names the pipeline expects will be duplicated from 'classes'
        self._label_fields: List[str] = list(getattr(bp, "label_fields", [])) if bp else []
        if self.classes_field and self.classes_field not in self._label_fields:
            # Keep our own name too so we can read back labels
            self._label_fields.append(self.classes_field)

    def _frame_kwargs(
        self,
        images_u8: np.ndarray,
        idx: int,
        keypoints_seq: Optional[List[List[Tuple[float, float]]]],
        bboxes_seq: Optional[List[List[Tuple[float, float, float, float]]]],
        classes_seq: Optional[List[List[int]]],
    ) -> Dict[str, Any]:
        img = images_u8[idx]  # (H,W,C) uint8
        H, W = img.shape[:2]
        kwargs: Dict[str, Any] = {"image": img}

        if self._has_keypoints:
            kps = keypoints_seq[idx] if keypoints_seq is not None else []
            kwargs[self.keypoints_field] = kps

        if self._has_bboxes:
            boxes_in = bboxes_seq[idx] if bboxes_seq is not None else []
            labels_in = classes_seq[idx] if classes_seq is not None else []
            boxes_out, labels_out = _sanitize_coco_boxes(boxes_in, labels_in, (H, W))
            kwargs[self.bboxes_field] = boxes_out
            for name in self._label_fields:
                kwargs[name] = labels_out

        return kwargs

    def __call__(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        if "inputs" not in sample:
            raise KeyError("Sample must include 'inputs'.")

        images_u8 = _tchw_to_hwc_u8(sample["inputs"])
        T = images_u8.shape[0]

        targets_in = dict(sample.get("targets", {}))

        # Prepare per-frame sequences
        keypoints_seq: Optional[List[List[Tuple[float, float]]]] = None
        if self._has_keypoints:
            raw_kp = _as_seq(targets_in.get(self.keypoints_field))
            keypoints_seq = (
                [_flatten_keypoints_frame(fr) for fr in raw_kp] if raw_kp is not None else [[] for _ in range(T)]
            )

        bboxes_seq: Optional[List[List[Tuple[float, float, float, float]]]] = None
        classes_seq: Optional[List[List[int]]] = None
        if self._has_bboxes:
            raw_b = _as_seq(targets_in.get(self.bboxes_field)) or [[] for _ in range(T)]
            raw_c = _as_seq(targets_in.get(self.classes_field)) or [[] for _ in range(T)]
            bboxes_seq = [_convert_bboxes_frame(fr) for fr in raw_b]
            classes_seq = [_convert_classes_frame(fr) for fr in raw_c]

        # First frame (capture replay if pipeline supports it)
        first_kwargs = self._frame_kwargs(images_u8, 0, keypoints_seq, bboxes_seq, classes_seq)
        first_out = self.pipeline(**first_kwargs)
        replay = first_out.get("replay") if isinstance(self.pipeline, A.ReplayCompose) else None

        # Accumulate outputs
        imgs_out = [_to_chw_f32(first_out["image"])]
        kps_out = [first_out.get(self.keypoints_field, [])] if self._has_keypoints else None
        bxs_out = [first_out.get(self.bboxes_field, [])] if self._has_bboxes else None

        def _extract_labels(d: Dict[str, Any]) -> List[int]:
            for name in self._label_fields:
                if name in d:
                    return list(map(int, d[name]))
            return []

        cls_out = [_extract_labels(first_out)] if self._has_bboxes else None

        for i in range(1, T):
            kwargs = self._frame_kwargs(images_u8, i, keypoints_seq, bboxes_seq, classes_seq)
            out = A.ReplayCompose.replay(replay, **kwargs) if replay is not None else self.pipeline(**kwargs)

            imgs_out.append(_to_chw_f32(out["image"]))
            if kps_out is not None:
                kps_out.append(out.get(self.keypoints_field, []))
            if bxs_out is not None:
                bxs_out.append(out.get(self.bboxes_field, []))
            if cls_out is not None:
                cls_out.append(_extract_labels(out))

        # Pack result
        out_sample: Dict[str, Any] = dict(sample)
        out_sample["inputs"] = torch.stack(imgs_out, dim=0).contiguous()

        targets_out = dict(targets_in)
        if self._has_keypoints and kps_out is not None:
            targets_out[self.keypoints_field] = kps_out
        if self._has_bboxes and bxs_out is not None and cls_out is not None:
            targets_out[self.bboxes_field] = bxs_out
            targets_out[self.classes_field] = cls_out

        out_sample["targets"] = targets_out
        return out_sample


# ------------------------------ public factory ------------------------------


def make_clip_replay_adapter(
    pipeline: Any,
    *,
    keypoints_field: str = "keypoints",
    bboxes_field: str = "bboxes",
    classes_field: str = "classes",
) -> ClipReplayAdapter:
    _ensure_albu()
    return ClipReplayAdapter(
        pipeline,
        keypoints_field=keypoints_field,
        bboxes_field=bboxes_field,
        classes_field=classes_field,
    )


__all__ = ["AlbumentationsReplayUnavailable", "make_clip_replay_adapter", "ClipReplayAdapter"]
