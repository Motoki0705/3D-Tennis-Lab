from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import json
import os

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image

from development.ball_tracking.ball_heatmap.training.aug_seq import (
    ReplayCompose,
    default_weak_aug,
    default_strong_aug,
)


@dataclass
class DataConfig:
    images_root: str
    labeled_json: str
    unlabeled_json: str
    img_size: Tuple[int, int]
    T: int
    frame_stride: int
    scales: List[int]
    sigma_px: List[float]
    supervise_hm_on_v1: bool = False


def letterbox(img: torch.Tensor, target_h: int, target_w: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resize with aspect preserve + symmetric pad to target size.
    Returns (img_resized, affine) where affine maps original xy -> new xy.
    img: CHW uint8 or float in [0,1].
    """
    _, h, w = img.shape
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img_resized = TF.resize(img, [new_h, new_w])
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]
    img_padded = TF.pad(img_resized, pad, fill=0)
    # affine: x' = x*scale + pad_left; y' = y*scale + pad_top
    affine = torch.tensor([scale, pad[0], pad[1]], dtype=torch.float32)
    return img_padded, affine


def generate_gaussian_heatmap(h: int, w: int, cx: float, cy: float, sigma: float) -> torch.Tensor:
    y = torch.arange(h, dtype=torch.float32).view(-1, 1)
    x = torch.arange(w, dtype=torch.float32).view(1, -1)
    g = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    return g


def downscale_coord(x: float, y: float, stride: int) -> Tuple[float, float]:
    return x / stride, y / stride


class BallDataset(Dataset):
    """Sequence dataset for ball heatmap training.

    - Supports COCO-style annotation files with multiple categories (e.g., ball/player).
    - Robustly extracts the ball keypoint (x, y, v) by filtering annotations with
      category == "ball" (or falling back to category_id == 1 if names are absent).
    - Supports both `original_path` and `file_name` for image file resolution.
    - Groups frames into clips using (game_id, clip_id) when available; otherwise,
      groups by the parent directory of the image path.
    """

    def __init__(self, cfg: DataConfig, labeled: bool, semisup: bool):
        self.cfg = cfg
        self.labeled = labeled
        self.semisup = semisup
        self.data = self._load_json(cfg.labeled_json if labeled else cfg.unlabeled_json)

        # --- COCO Annotation Parsing (ball keypoint extraction) ---
        # Build a map of images for efficient lookup
        images_map: Dict[int, Dict[str, Any]] = {im["id"]: dict(im) for im in self.data.get("images", [])}

        # Extract the ball category id if possible
        ball_cat_id = self._get_ball_category_id(self.data)

        # Merge ball annotations into the image info dict (x, y, v)
        if self.labeled and self.data.get("annotations"):
            for ann in self.data["annotations"]:
                if ball_cat_id is not None and ann.get("category_id") != ball_cat_id:
                    continue  # skip non-ball annotations
                img_id = ann.get("image_id")
                if img_id not in images_map:
                    continue
                xyv = self._extract_ball_keypoint(ann)
                if xyv is not None:
                    x, y, v = xyv
                    images_map[img_id]["keypoints"] = [x, y]
                    images_map[img_id]["visibility"] = int(v)

        # Use the merged image list for grouping into clips
        self.clips = self._group_clips({"images": list(images_map.values())})
        # --- End of Parsing ---

        self.weak_aug = ReplayCompose(default_weak_aug)
        self.strong_aug = ReplayCompose(default_strong_aug)

        # Create a flat list of all possible samples (sequences) from all clips
        self.samples = []
        for clip_idx, clip in enumerate(self.clips):
            num_frames_in_clip = len(clip["frames"])
            T = self.cfg.T
            stride = self.cfg.frame_stride
            # The number of possible starting frames for a valid sequence
            num_possible_starts = num_frames_in_clip - (T - 1) * stride
            if num_possible_starts > 0:
                for start_frame_idx in range(num_possible_starts):
                    self.samples.append({
                        "clip_idx": clip_idx,
                        "start_frame": start_frame_idx,
                    })

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return json.load(f)

    def _group_clips(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Group frames into clips for sequential sampling.

        Priority of grouping keys:
        1) If all frames contain integer `game_id` and `clip_id`, group by these.
        2) Else, group by the parent directory of the image path (`original_path` or `file_name`).
        """
        images = data.get("images", [])

        # Decide grouping mode
        have_ids = all(("game_id" in im and "clip_id" in im) for im in images) and len(images) > 0

        clips: List[Dict[str, Any]] = []
        if have_ids:
            # group by (game_id, clip_id)
            groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
            for im in images:
                gid = int(im.get("game_id"))
                cid = int(im.get("clip_id"))
                groups.setdefault((gid, cid), []).append(im)
            for (gid, cid), frames in groups.items():
                frames_sorted = sorted(
                    frames,
                    key=lambda x: (
                        x.get("frame_id", -1),
                        self._safe_path(x).lower(),
                    ),
                )
                clips.append({"game_id": gid, "clip_id": cid, "frames": frames_sorted})
        else:
            # group by parent directory of the file path
            from collections import defaultdict

            groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for im in images:
                p = self._safe_path(im)
                parent = os.path.dirname(p)
                groups[parent].append(im)

            # Produce pseudo numeric ids for meta
            for idx, (parent, frames) in enumerate(sorted(groups.items())):
                frames_sorted = sorted(
                    frames,
                    key=lambda x: (
                        x.get("frame_id", -1),
                        self._natural_key(self._safe_path(x)),
                    ),
                )
                clips.append({"game_id": idx, "clip_id": 0, "frames": frames_sorted})

        return clips

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_info = self.samples[idx]
        clip_idx = sample_info["clip_idx"]
        start_frame = sample_info["start_frame"]

        clip = self.clips[clip_idx]

        T = self.cfg.T
        stride = self.cfg.frame_stride

        # Get the frames for the sequence based on start_frame, T, and stride
        frame_indices = range(start_frame, start_frame + T * stride, stride)
        frames = [clip["frames"][i] for i in frame_indices]

        # Get the next frame if it exists, to calculate speed for the last frame
        next_frame = None
        last_frame_index_in_clip = start_frame + (T - 1) * stride
        if last_frame_index_in_clip + stride < len(clip["frames"]):
            next_frame = clip["frames"][last_frame_index_in_clip + stride]

        imgs: List[torch.Tensor] = []
        affines: List[torch.Tensor] = []
        xs, ys, vs = [], [], []
        for f in frames:
            path_rel = f.get("original_path") or f.get("file_name")
            path = path_rel if os.path.isabs(path_rel) else os.path.join(self.cfg.images_root, path_rel)
            with Image.open(path) as im:
                im = im.convert("RGB")
                img = TF.pil_to_tensor(im).float() / 255.0  # CHW
            img, affine = letterbox(img, self.cfg.img_size[0], self.cfg.img_size[1])
            affines.append(affine)
            imgs.append(img)

            kp = f.get("keypoints", None)
            v = int(f.get("visibility", 0))
            if kp is not None and len(kp) >= 2:
                x, y = float(kp[0]), float(kp[1])
                x = x * affine[0].item() + affine[1].item()
                y = y * affine[0].item() + affine[2].item()
            else:
                x, y = float("nan"), float("nan")
            xs.append(x)
            ys.append(y)
            vs.append(v)

        video = torch.stack(imgs, dim=0)  # T,C,H,W

        # Get coordinates of the next frame
        next_xy = None
        if next_frame and len(affines) > 0:
            kp = next_frame.get("keypoints", None)
            if kp is not None and len(kp) >= 2:
                # Use the last frame's affine (assuming consistent source resolution within a clip)
                last_affine = affines[-1]
                x, y = float(kp[0]), float(kp[1])
                x = x * last_affine[0].item() + last_affine[1].item()
                y = y * last_affine[0].item() + last_affine[2].item()
                next_xy = (x, y)

        sup_dict: Dict[str, Any] = {
            "video": video,
            "targets": self._make_targets(xs, ys, vs, next_xy),
            "meta": {
                "game_id": clip["game_id"],
                "clip_id": clip["clip_id"],
                "paths": [self._safe_path(f) for f in frames],
            },
        }

        sample: Dict[str, Any] = {"sup": sup_dict}

        if self.semisup and not self.labeled:
            weak = self.weak_aug([f for f in video])
            strong = self.strong_aug([f for f in video])
            weak_t = torch.stack(weak, dim=0)
            strong_t = torch.stack(strong, dim=0)
            sample["unsup"] = {
                "weak": weak_t,
                "strong": strong_t,
                "meta": sup_dict["meta"],
            }

        return sample

    def _calculate_speed(
        self, xs: List[float], ys: List[float], vs: List[int], H: int, W: int, next_xy: Optional[Tuple[float, float]]
    ) -> torch.Tensor:
        """Calculates speed using forward difference (position(t+1) - position(t))."""
        import math

        T = len(xs)
        speed = torch.zeros(T, 2, dtype=torch.float32)

        # Append the next frame's coordinates to the list for easier processing
        all_xs = xs + ([next_xy[0]] if next_xy else [float("nan")])
        all_ys = ys + ([next_xy[1]] if next_xy else [float("nan")])

        # First, calculate speeds where possible using forward difference
        invW = 1.0 / max(float(W), 1.0)
        invH = 1.0 / max(float(H), 1.0)
        for t in range(T):
            # Speed at t is valid if position at t and t+1 are valid
            if not (
                math.isnan(all_xs[t]) or math.isnan(all_xs[t + 1]) or math.isnan(all_ys[t]) or math.isnan(all_ys[t + 1])
            ):
                dx = (all_xs[t + 1] - all_xs[t]) * invW
                dy = (all_ys[t + 1] - all_ys[t]) * invH
                speed[t, 0] = dx
                speed[t, 1] = dy
            else:
                speed[t, :] = float("nan")

        # Second, back-fill NaN values with the next valid speed
        last_valid_speed = torch.tensor([0.0, 0.0])  # Default speed if no future speed is valid
        for t in range(T - 1, -1, -1):
            if torch.isnan(speed[t]).any():
                speed[t] = last_valid_speed
            else:
                last_valid_speed = speed[t].clone()

        return speed

    def _make_targets(
        self, xs: List[float], ys: List[float], vs: List[int], next_xy: Optional[Tuple[float, float]]
    ) -> Dict[str, Any]:
        T = len(xs)
        H, W = self.cfg.img_size

        # Calculate speed using the dedicated function with forward difference
        speed = self._calculate_speed(xs, ys, vs, H, W, next_xy)

        # vis_state (0/1/2)
        vis_state = torch.tensor(vs, dtype=torch.long)
        # masks
        vis_mask_hm = (vis_state == 2) | ((vis_state == 1) & self.cfg.supervise_hm_on_v1)
        vis_mask_speed = torch.ones(T, dtype=torch.bool)

        # heatmaps per scale
        hm_list: List[torch.Tensor] = []
        for stride, sigma in zip(self.cfg.scales, self.cfg.sigma_px):
            h_s, w_s = H // stride, W // stride
            heat = torch.zeros(T, 1, h_s, w_s, dtype=torch.float32)
            for t in range(T):
                if not vis_mask_hm[t]:
                    continue
                x, y = xs[t], ys[t]
                if not (x == x and y == y):  # check for NaN
                    continue
                cx, cy = downscale_coord(x, y, stride)
                g = generate_gaussian_heatmap(h_s, w_s, cx, cy, sigma / stride)
                heat[t, 0] = g
            hm_list.append(heat)

        targets = {
            "hm": hm_list,  # list per scale [T,1,Hs,Ws]
            "speed": speed,  # [T,2]
            "vis_state": vis_state,  # [T]
            "vis_mask_hm": vis_mask_hm,  # [T]
            "vis_mask_speed": vis_mask_speed,  # [T]
            "coords_img": torch.tensor(list(zip(xs, ys)), dtype=torch.float32),  # [T, 2]
        }
        return targets

    # --- Helpers ---
    def _get_ball_category_id(self, data: Dict[str, Any]) -> Optional[int]:
        cats = data.get("categories", [])
        if not cats:
            return 1  # common default in our repos
        # Prefer name match
        for c in cats:
            if str(c.get("name", "")).lower() == "ball":
                return int(c.get("id", 1))
        # Fallback: first category with exactly one keypoint
        for c in cats:
            kps = c.get("keypoints")
            if isinstance(kps, list) and len(kps) == 1:
                return int(c.get("id", 1))
        # Fallback to 1
        return 1

    def _extract_ball_keypoint(self, ann: Dict[str, Any]) -> Optional[Tuple[float, float, int]]:
        kps = ann.get("keypoints")
        if isinstance(kps, list) and len(kps) >= 3:
            x, y, v = kps[0], kps[1], kps[2]
            try:
                return float(x), float(y), int(v)
            except Exception:
                return None
        # Optional: derive center from bbox if available
        bbox = ann.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            x, y, w, h = bbox
            try:
                cx = float(x) + float(w) / 2.0
                cy = float(y) + float(h) / 2.0
                return cx, cy, int(ann.get("num_keypoints", 0) > 0) * 2
            except Exception:
                return None
        return None

    def _safe_path(self, im: Dict[str, Any]) -> str:
        return im.get("original_path") or im.get("file_name") or ""

    def _natural_key(self, s: str):
        """Sort helper that treats digits as numbers (e.g., img2 < img10)."""
        import re

        return [int(t) if t.isdigit() else t for t in re.findall(r"\d+|\D+", s)]


if __name__ == "__main__":
    # Example usage
    cfg = DataConfig(
        images_root=r"data\processed\ball\images",
        labeled_json=r"data\processed\ball\annotation.json",
        unlabeled_json=r"data\processed\ball\non_annotation.json",
        img_size=(320, 640),
        T=5,
        frame_stride=1,
        scales=[4, 8, 16],
        sigma_px=[2.0, 4.0, 8.0],
        supervise_hm_on_v1=True,
    )
    dataset = BallDataset(cfg, labeled=True, semisup=True)
    sample = dataset[0]
    print(sample)
    print("Supervised video shape:", sample["sup"]["video"].shape)
    if "unsup" in sample:
        print("Unsupervised weak shape:", sample["unsup"]["weak"].shape)
        print("Unsupervised strong shape:", sample["unsup"]["strong"].shape)
