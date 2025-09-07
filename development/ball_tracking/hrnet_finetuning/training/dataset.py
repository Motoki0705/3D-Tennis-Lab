from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Callable

import json
import os

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A


@dataclass
class DataConfig:
    images_root: str
    labeled_json: str
    img_size: Tuple[int, int]
    T: int
    frame_stride: int
    output_stride: int
    sigma_px: float


def generate_gaussian_heatmap(h: int, w: int, cx: float, cy: float, sigma: float) -> torch.Tensor:
    y = torch.arange(h, dtype=torch.float32).view(-1, 1)
    x = torch.arange(w, dtype=torch.float32).view(1, -1)
    g = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    return g


def downscale_coord(x: float, y: float, stride: int) -> Tuple[float, float]:
    return x / stride, y / stride


class BallDataset(Dataset):
    """Sequence dataset for ball heatmap training."""

    def __init__(
        self,
        cfg: DataConfig,
        transforms: Optional[Callable] = None,
        *,
        allowed_clip_indices: Optional[List[int]] = None,
        data_override: Optional[Dict[str, Any]] = None,
    ):
        self.cfg = cfg
        self.transforms = transforms
        # Allow injecting preloaded/filtered data to avoid IO and enable splitting
        self.data = data_override if data_override is not None else self._load_json(cfg.labeled_json)

        images_map: Dict[int, Dict[str, Any]] = {im["id"]: dict(im) for im in self.data.get("images", [])}
        ball_cat_id = self._get_ball_category_id(self.data)

        if self.data.get("annotations"):
            for ann in self.data["annotations"]:
                if ball_cat_id is not None and ann.get("category_id") != ball_cat_id:
                    continue
                img_id = ann.get("image_id")
                if img_id not in images_map:
                    continue
                xyv = self._extract_ball_keypoint(ann)
                if xyv is not None:
                    images_map[img_id]["keypoints"] = [xyv[0], xyv[1]]

        clips_all = self._group_clips({"images": list(images_map.values())})
        if allowed_clip_indices is not None:
            self.clips = [clips_all[i] for i in allowed_clip_indices if 0 <= i < len(clips_all)]
        else:
            self.clips = clips_all

        self.samples = []
        for clip_idx, clip in enumerate(self.clips):
            num_frames_in_clip = len(clip["frames"])
            num_possible_starts = num_frames_in_clip - (self.cfg.T - 1) * self.cfg.frame_stride
            if num_possible_starts > 0:
                for start_frame_idx in range(num_possible_starts):
                    self.samples.append({"clip_idx": clip_idx, "start_frame": start_frame_idx})

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return json.load(f)

    def _group_clips(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        images = data.get("images", [])
        have_ids = all(("game_id" in im and "clip_id" in im) for im in images) and len(images) > 0
        clips: List[Dict[str, Any]] = []
        if have_ids:
            groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
            for im in images:
                groups.setdefault((int(im.get("game_id")), int(im.get("clip_id"))), []).append(im)
            for (gid, cid), frames in groups.items():
                frames_sorted = sorted(frames, key=lambda x: (x.get("frame_id", -1), self._safe_path(x).lower()))
                clips.append({"game_id": gid, "clip_id": cid, "frames": frames_sorted})
        else:
            from collections import defaultdict

            groups = defaultdict(list)
            for im in images:
                groups[os.path.dirname(self._safe_path(im))].append(im)
            for idx, (parent, frames) in enumerate(sorted(groups.items())):
                frames_sorted = sorted(
                    frames, key=lambda x: (x.get("frame_id", -1), self._natural_key(self._safe_path(x)))
                )
                clips.append({"game_id": idx, "clip_id": 0, "frames": frames_sorted})
        return clips

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_info = self.samples[idx]
        clip = self.clips[sample_info["clip_idx"]]
        start_frame = sample_info["start_frame"]

        frame_indices = range(start_frame, start_frame + self.cfg.T * self.cfg.frame_stride, self.cfg.frame_stride)
        frames_meta = [clip["frames"][i] for i in frame_indices]

        images_np, keypoints_per_frame = [], []
        for f_meta in frames_meta:
            path = self._safe_path(f_meta)
            path = path if os.path.isabs(path) else os.path.join(self.cfg.images_root, path)
            with Image.open(path) as im:
                images_np.append(np.array(im.convert("RGB")))
            kp = f_meta.get("keypoints")
            keypoints_per_frame.append([(float(kp[0]), float(kp[1]))] if kp and len(kp) >= 2 else [])

        transformed_imgs, transformed_kps = [], []
        if self.transforms:
            first_frame_data = self.transforms(image=images_np[0], keypoints=keypoints_per_frame[0])
            transformed_imgs.append(first_frame_data["image"])
            transformed_kps.append(first_frame_data["keypoints"])
            if len(images_np) > 1:
                replay = first_frame_data.get("replay")
                # Sanitize replay to avoid Albumentations warnings with PadIfNeeded('padding') on some versions
                if isinstance(replay, dict):
                    try:
                        for t in replay.get("transforms", []):
                            args = t.get("args", {})
                            if isinstance(args, dict) and "padding" in args:
                                args.pop("padding", None)
                    except Exception:
                        pass
                for i in range(1, len(images_np)):
                    data = (
                        A.ReplayCompose.replay(replay, image=images_np[i], keypoints=keypoints_per_frame[i])
                        if replay
                        else self.transforms(image=images_np[i], keypoints=keypoints_per_frame[i])
                    )
                    transformed_imgs.append(data["image"])
                    transformed_kps.append(data["keypoints"])
            video = torch.stack(transformed_imgs, dim=0)
        else:
            video = torch.stack([torch.from_numpy(img).permute(2, 0, 1) for img in images_np], dim=0).float() / 255.0

        video = video.permute(1, 0, 2, 3)

        H, W = self.cfg.img_size
        h_s, w_s = H // self.cfg.output_stride, W // self.cfg.output_stride
        heatmap = torch.zeros(1, h_s, w_s, dtype=torch.float32)

        last_kps = transformed_kps[-1]
        if last_kps:
            x, y = last_kps[0]
            if np.isfinite(x) and np.isfinite(y) and 0 <= x < W and 0 <= y < H:
                cx, cy = downscale_coord(x, y, self.cfg.output_stride)
                heatmap[0] = generate_gaussian_heatmap(h_s, w_s, cx, cy, self.cfg.sigma_px)

        return video, heatmap

    def _get_ball_category_id(self, data: Dict[str, Any]) -> Optional[int]:
        cats = data.get("categories", [])
        if not cats:
            return 1
        for c in cats:
            if str(c.get("name", "")).lower() == "ball":
                return int(c.get("id", 1))
        return 1

    def _extract_ball_keypoint(self, ann: Dict[str, Any]) -> Optional[Tuple[float, float, int]]:
        kps = ann.get("keypoints")
        if isinstance(kps, list) and len(kps) >= 3:
            try:
                return float(kps[0]), float(kps[1]), int(kps[2])
            except Exception:
                return None
        bbox = ann.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                return (
                    float(bbox[0]) + float(bbox[2]) / 2.0,
                    float(bbox[1]) + float(bbox[3]) / 2.0,
                    int(ann.get("num_keypoints", 0) > 0) * 2,
                )
            except Exception:
                return None
        return None

    def _safe_path(self, im: Dict[str, Any]) -> str:
        return im.get("original_path") or im.get("file_name") or ""

    def _natural_key(self, s: str):
        import re

        return [int(t) if t.isdigit() else t for t in re.findall(r"\d+|\D+", s)]
