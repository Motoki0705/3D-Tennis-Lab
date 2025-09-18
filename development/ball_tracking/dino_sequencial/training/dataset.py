from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
import collections

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F


if hasattr(Image, "Resampling"):
    _RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
else:  # pragma: no cover - backwards compatibility
    _RESAMPLE_BICUBIC = Image.BICUBIC


def _gaussian_2d(h: int, w: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
    return g.astype(np.float32)


def _load_sequences(path: Path) -> Dict[str, List[Dict]]:
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    print(f"--- Loading COCO annotations for sequences from: {path} ---")

    with open(path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # 1. Process annotations first to map image_id to keypoints
    image_id_to_centers = collections.defaultdict(list)
    for ann in coco_data.get("annotations", []):
        keypoints = ann.get("keypoints")
        if keypoints and len(keypoints) >= 3 and keypoints[2] > 0:
            image_id_to_centers[ann["image_id"]].append(keypoints[:2])

    # 2. Group frames by (game_id, clip_id)
    video_frames: Dict[str, List[Dict]] = collections.defaultdict(list)
    for img_info in coco_data.get("images", []):
        game_id = img_info.get("game_id")
        clip_id = img_info.get("clip_id")

        if game_id is None or clip_id is None:
            continue

        video_id = f"game_{game_id}_clip_{clip_id}"

        frame_data = {"file_name": img_info["file_name"], "original_path": img_info["original_path"]}

        centers = image_id_to_centers.get(img_info["id"])
        if centers:
            # For simplicity, take the first valid annotation if multiple exist
            frame_data["center"] = centers[0]

        video_frames[video_id].append(frame_data)

    # 3. Sort frames within each video by file_name
    for video_id in video_frames:
        video_frames[video_id].sort(key=lambda x: x["file_name"])

    print(f"--- Found {len(video_frames)} videos. ---")
    return video_frames


@dataclass
class DatasetConfig:
    images_root: str
    labeled_json: str
    img_size: Tuple[int, int] = (640, 640)
    output_stride: int = 4
    sigma_px: float = 2.0
    sequence_length: int = 8


class BallSequenceDataset(Dataset):
    def __init__(
        self,
        cfg: DatasetConfig,
    ) -> None:
        super().__init__()
        self.images_root = Path(cfg.images_root)
        self.sequence_length = cfg.sequence_length

        video_sequences = _load_sequences(Path(cfg.labeled_json))

        self.items = []
        for video_id, frames in video_sequences.items():
            if len(frames) >= self.sequence_length:
                for i in range(len(frames) - self.sequence_length + 1):
                    self.items.append(frames[i : i + self.sequence_length])

        self.img_size = tuple(cfg.img_size)
        self.out_stride = int(cfg.output_stride)
        self.sigma_px = float(cfg.sigma_px)
        self.default_long_side = max(self.img_size)

        self.normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self._index_long_side: Dict[int, int] = {}
        self._shared_long_side_map = None

    def __len__(self) -> int:
        return len(self.items)

    def _resolve_path(self, root: Path, p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else (root / q)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_sequence = self.items[idx]

        first_frame = frame_sequence[0]
        first_img_path = self._resolve_path(self.images_root, first_frame["original_path"]).resolve()
        first_img = Image.open(first_img_path).convert("RGB")
        first_orig_w, first_orig_h = first_img.size
        long_side_override = self._get_long_side(idx)
        desired_long_side = self._resolve_target_long_side(long_side_override)
        target_h = target_w = desired_long_side

        images = []
        heatmaps = []

        for i, frame_data in enumerate(frame_sequence):
            if i == 0:
                img = first_img
                orig_w, orig_h = first_orig_w, first_orig_h
            else:
                img_path = self._resolve_path(self.images_root, frame_data["original_path"]).resolve()
                img = Image.open(img_path).convert("RGB")
                orig_w, orig_h = img.size

            resize_params = self._compute_resize_params(orig_h, orig_w, desired_long_side)
            resized_w, resized_h = resize_params["resize_wh"]
            padding = resize_params["padding"]
            scale_h, scale_w = resize_params["scale_hw"]
            if (resized_w, resized_h) != img.size:
                img = img.resize((resized_w, resized_h), resample=_RESAMPLE_BICUBIC)
            img_tensor = F.to_tensor(img)
            pad_left, pad_right, pad_top, pad_bottom = padding
            if pad_left or pad_right or pad_top or pad_bottom:
                img_tensor = F.pad(img_tensor, (pad_left, pad_right, pad_top, pad_bottom))
            img_tensor = self.normalize(img_tensor)
            images.append(img_tensor)

            H, W = img_tensor.shape[-2], img_tensor.shape[-1]
            h_out, w_out = H // self.out_stride, W // self.out_stride

            center = frame_data.get("center")
            if center is None:
                heatmap = torch.zeros((1, h_out, w_out), dtype=torch.float32)
            else:
                cx, cy = float(center[0]), float(center[1])
                cx_resized = cx * scale_w + pad_left
                cy_resized = cy * scale_h + pad_top
                sx = (w_out - 1) / max(1, W - 1)
                sy = (h_out - 1) / max(1, H - 1)
                cx_hm = cx_resized * sx
                cy_hm = cy_resized * sy
                g = _gaussian_2d(h_out, w_out, cx_hm, cy_hm, sigma=self.sigma_px)
                heatmap = torch.from_numpy(g).unsqueeze(0)

            heatmaps.append(heatmap)

        # Stack into (T, C, H, W) and (T, 1, H_out, W_out)
        return torch.stack(images), torch.stack(heatmaps)

    def set_shared_long_side_map(self, shared_map) -> None:
        self._shared_long_side_map = shared_map

    def set_long_side_map(self, mapping: Mapping[int, int]) -> None:
        if self._shared_long_side_map is not None:
            self._shared_long_side_map.clear()
            self._shared_long_side_map.update(mapping)
        else:
            self._index_long_side = dict(mapping)

    def clear_long_side_map(self) -> None:
        if self._shared_long_side_map is not None:
            self._shared_long_side_map.clear()
        else:
            self._index_long_side = {}

    def _get_long_side(self, idx: int) -> Optional[int]:
        if self._shared_long_side_map is not None:
            return self._shared_long_side_map.get(idx)
        return self._index_long_side.get(idx)

    def _resolve_target_long_side(self, long_side_override: Optional[int]) -> int:
        desired = long_side_override if long_side_override is not None else self.default_long_side
        desired = max(16, int(np.round(desired / 16.0) * 16))
        return desired

    def _compute_resize_params(
        self,
        orig_h: int,
        orig_w: int,
        desired_long: int,
    ) -> Dict[str, Any]:
        target_h = desired_long
        target_w = desired_long

        if orig_h <= 0 or orig_w <= 0:
            return {
                "target_shape": (target_h, target_w),
                "resize_wh": (target_w, target_h),
                "padding": (0, 0, 0, 0),
                "scale_hw": (1.0, 1.0),
            }

        scale = min(target_h / float(orig_h), target_w / float(orig_w))
        resized_h = int(np.round(orig_h * scale / 16.0) * 16)
        resized_w = int(np.round(orig_w * scale / 16.0) * 16)
        resized_h = int(np.clip(resized_h, 16, target_h))
        resized_w = int(np.clip(resized_w, 16, target_w))

        pad_vert = max(0, target_h - resized_h)
        pad_h_top = pad_vert // 2
        pad_h_bottom = pad_vert - pad_h_top
        pad_h_top = int(pad_h_top)
        pad_h_bottom = int(pad_h_bottom)

        pad_horiz = max(0, target_w - resized_w)
        pad_w_left = pad_horiz // 2
        pad_w_right = pad_horiz - pad_w_left
        pad_w_left = int(pad_w_left)
        pad_w_right = int(pad_w_right)

        scale_h = resized_h / max(1, orig_h)
        scale_w = resized_w / max(1, orig_w)

        return {
            "target_shape": (target_h, target_w),
            "resize_wh": (resized_w, resized_h),
            "padding": (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom),
            "scale_hw": (scale_h, scale_w),
        }


__all__ = ["DatasetConfig", "BallSequenceDataset"]
