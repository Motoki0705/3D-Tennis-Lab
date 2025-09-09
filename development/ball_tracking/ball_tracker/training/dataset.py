from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import json
import os
import torch
from torch.utils.data import Dataset
import numpy as np


@dataclass
class DataConfig:
    labeled_json: str
    sequence_length: int
    predict_offset: int


class BallVectorDataset(Dataset):
    """Dataset to provide sequences of ball state vectors (pos, vel, acc)."""

    def __init__(
        self,
        cfg: DataConfig,
        allowed_clip_indices: Optional[List[int]] = None,
        data_override: Optional[Dict[str, Any]] = None,
    ):
        self.cfg = cfg
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
                    # Store as (x, y, visible)
                    images_map[img_id]["keypoints"] = [xyv[0], xyv[1], xyv[2]]

        clips_all = self._group_clips({"images": list(images_map.values())})
        if allowed_clip_indices is not None:
            self.clips = [clips_all[i] for i in allowed_clip_indices if 0 <= i < len(clips_all)]
        else:
            self.clips = clips_all

        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._process_clips()

    def _process_clips(self):
        """Pre-processes all clips to generate state vectors and create samples."""
        for clip in self.clips:
            # 1. Extract coordinates and visibility
            coords = []
            for frame in clip["frames"]:
                kp = frame.get("keypoints")
                if kp and len(kp) >= 3:
                    coords.append([kp[0], kp[1], float(kp[2] > 0)])  # x, y, is_visible
                else:
                    coords.append([np.nan, np.nan, 0.0])
            coords_np = np.array(coords, dtype=np.float32)

            # 2. Interpolate missing points
            coords_np = self._interpolate_nan(coords_np)

            # 3. Calculate velocity and acceleration
            # Velocity (v = p_t - p_{t-1})
            velocity = np.zeros_like(coords_np[:, :2])
            velocity[1:] = coords_np[1:, :2] - coords_np[:-1, :2]
            # Acceleration (a = v_t - v_{t-1})
            acceleration = np.zeros_like(velocity)
            acceleration[1:] = velocity[1:] - velocity[:-1]

            # 4. Combine into state vectors [x, y, vx, vy, ax, ay]
            state_vectors = np.hstack([coords_np[:, :2], velocity, acceleration])
            state_tensors = torch.from_numpy(state_vectors).float()

            # 5. Create samples
            total_len = len(state_tensors)
            seq_len = self.cfg.sequence_length
            offset = self.cfg.predict_offset
            for i in range(total_len - seq_len - offset + 1):
                input_seq = state_tensors[i : i + seq_len]
                target_vec = state_tensors[i + seq_len + offset - 1]
                self.samples.append((input_seq, target_vec))

    def _interpolate_nan(self, arr: np.ndarray) -> np.ndarray:
        """Linearly interpolates NaN values in the x, y coordinates."""
        for i in range(arr.shape[1]):  # Iterate over x, y
            nans = np.isnan(arr[:, i])
            if not np.any(nans):
                continue
            x = lambda z: z.nonzero()[0]
            arr[nans, i] = np.interp(x(nans), x(~nans), arr[~nans, i])
        return arr

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return json.load(f)

    def _group_clips(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        # This function is copied from the reference implementation
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

    def _get_ball_category_id(self, data: Dict[str, Any]) -> Optional[int]:
        # This function is copied from the reference implementation
        cats = data.get("categories", [])
        if not cats:
            return 1
        for c in cats:
            if str(c.get("name", "")).lower() == "ball":
                return int(c.get("id", 1))
        return 1

    def _extract_ball_keypoint(self, ann: Dict[str, Any]) -> Optional[Tuple[float, float, int]]:
        # This function is copied from the reference implementation
        kps = ann.get("keypoints")
        if isinstance(kps, list) and len(kps) >= 3:
            return float(kps[0]), float(kps[1]), int(kps[2])
        bbox = ann.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            return (
                float(bbox[0]) + float(bbox[2]) / 2.0,
                float(bbox[1]) + float(bbox[3]) / 2.0,
                int(ann.get("num_keypoints", 0) > 0) * 2,
            )
        return None

    def _safe_path(self, im: Dict[str, Any]) -> str:
        return im.get("original_path") or im.get("file_name") or ""

    def _natural_key(self, s: str):
        import re

        return [int(t) if t.isdigit() else t for t in re.findall(r"\d+|\D+", s)]
