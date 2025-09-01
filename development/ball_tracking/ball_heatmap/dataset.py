from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

try:  # optional Hydra dependency for absolute paths
    from hydra.utils import to_absolute_path as _to_abs
except Exception:  # pragma: no cover - simple fallback when Hydra not installed

    def _to_abs(p: str) -> str:
        from pathlib import Path

        return str(Path(p).absolute())


@dataclass
class HeatmapSpec:
    stride: int
    sigma: float


def _default_transforms(img_size: Tuple[int, int]):
    return A.Compose(
        [
            A.Resize(img_size[0], img_size[1]),
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def _natural_index(s: str) -> int:
    m = re.findall(r"(\d+)", s)
    return int(m[-1]) if m else 0


class BallDataset(Dataset):
    """
    Unified dataset (sequence-aware). When sequence_length == 1, behaves like the old single-frame dataset.

    - Groups images by parent directory to form clips; builds sequence centers.
    - Derives version from keypoints visibility `v` in annotations (v1/v2/...); unknown if missing.
    - Generates multiscale Gaussian heatmaps and 2ch offsets per frame.

    Returns when sequence_length > 1:
      - image: [T,3,H,W]
      - heatmaps: list of [T,1,Hs,Ws]
      - offsets: list of [T,2,Hs,Ws]
      - coord: [T,2]
      - valid_mask: [T]

    Returns when sequence_length == 1 (compat mode):
      - image: [3,H,W]
      - heatmaps: list of [1,Hs,Ws]
      - offsets: list of [2,Hs,Ws]
      - coord: [2]
      - valid_mask: scalar float tensor 0/1
    """

    def __init__(
        self,
        img_dir: Path,
        annotation_file: Path,
        img_size: Tuple[int, int],
        heatmap_specs: List[HeatmapSpec],
        sequence_length: int = 1,
        frame_stride: int = 1,
        center_version: Optional[str] = None,
        center_span: int = 1,
        negatives: str = "use",  # none|use|skip
        ball_category_id: int = 1,
        multi_ball_policy: str = "first",  # first|warn|error
        transform: Optional[A.Compose] = None,
    ):
        super().__init__()
        assert sequence_length >= 1 and sequence_length % 2 == 1, "sequence_length should be odd and >=1"
        assert center_span >= 1 and center_span % 2 == 1, "center_span should be odd and >=1"
        self.img_dir = Path(img_dir)
        self.img_size = tuple(img_size)
        self.heatmap_specs = heatmap_specs
        self.negatives = negatives
        self.L = int(sequence_length)
        self.r = self.L // 2
        self.frame_stride = int(frame_stride)
        self.center_version = center_version  # 'v1' or 'v2' or None
        self.center_span = int(center_span)
        self.ball_category_id = int(ball_category_id)
        self.multi_ball_policy = str(multi_ball_policy)

        self._transform = None
        self.transform = transform or _default_transforms(self.img_size)

        with open(_to_abs(str(annotation_file)), "r") as f:
            coco = json.load(f)

        # Per-image meta
        self.image_meta: Dict[int, Dict[str, Any]] = {int(img["id"]): img for img in coco["images"]}

        # Collect ball annotations per image (handle multiple by policy)
        anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
        for ann in coco.get("annotations", []):
            if int(ann.get("category_id", -1)) != self.ball_category_id:
                continue
            img_id = int(ann["image_id"])
            anns_by_img.setdefault(img_id, []).append(ann)

        import warnings

        ball_anns: Dict[int, Optional[Dict[str, Any]]] = {}
        for img_id, lst in anns_by_img.items():
            if len(lst) <= 1 or self.multi_ball_policy == "first":
                ball_anns[img_id] = lst[0]
            elif self.multi_ball_policy == "warn":
                warnings.warn(f"Multiple ball annotations for image_id={img_id}; using the first.")
                ball_anns[img_id] = lst[0]
            elif self.multi_ball_policy == "error":
                raise ValueError(f"Multiple ball annotations for image_id={img_id} not allowed (policy=error)")
            else:
                ball_anns[img_id] = lst[0]

        # Group images by (game_id, clip_id) if present; else fallback to path parent
        groups: Dict[str, List[int]] = {}
        for img_id, meta in self.image_meta.items():
            if "game_id" in meta and "clip_id" in meta:
                gkey = f"game{meta['game_id']}_clip{meta['clip_id']}"
            else:
                src = str(meta.get("original_path") or meta.get("file_name", ""))
                gkey = str(Path(src).parent)
            groups.setdefault(gkey, []).append(img_id)

        def _name_for_sort(mid: int) -> str:
            m = self.image_meta[mid]
            # Prefer file_name basename; fallback to original_path basename
            fn = str(m.get("file_name", ""))
            if fn:
                return str(Path(fn).name)
            op = str(m.get("original_path", ""))
            return str(Path(op).name)

        for g, ids in groups.items():
            ids.sort(key=lambda i: _natural_index(_name_for_sort(i)))

        # Build frames with version from keypoints visibility `v`
        self.groups: List[Dict[str, Any]] = []
        for g, ids in groups.items():
            frames: List[Dict[str, Any]] = []
            for img_id in ids:
                meta = self.image_meta[img_id]
                ann = ball_anns.get(img_id, None)
                if ann is not None and isinstance(ann.get("keypoints"), (list, tuple)) and len(ann["keypoints"]) >= 3:
                    try:
                        v = int(float(ann["keypoints"][2]))
                        version = f"v{v}"
                    except Exception:
                        version = "unknown"
                else:
                    version = "unknown"
                frames.append({"image_id": img_id, "meta": meta, "ann": ann, "version": version})
            self.groups.append({"group": g, "frames": frames})

        # Fast group lookup map
        self._group_map: Dict[str, Dict[str, Any]] = {g["group"]: g for g in self.groups}

        # Build sequence index centered on desired version (if any)
        self.index: List[Dict[str, Any]] = []
        span_r = self.center_span // 2
        for grp in self.groups:
            frames = grp["frames"]
            n = len(frames)
            for i in range(n):
                start = i - self.r * self.frame_stride
                end = i + self.r * self.frame_stride
                if start < 0 or end >= n:
                    continue
                if self.center_version is not None:
                    if frames[i]["version"] != self.center_version:
                        continue
                    if self.center_span > 1:
                        ok = True
                        for j in range(
                            i - span_r * self.frame_stride, i + span_r * self.frame_stride + 1, self.frame_stride
                        ):
                            if frames[j]["version"] != self.center_version:
                                ok = False
                                break
                        if not ok:
                            continue
                if frames[i]["ann"] is None and self.negatives == "skip":
                    continue
                self.index.append({"group": grp["group"], "center": i})

        # Version label per sample = center frame's version
        self.versions: List[str] = []
        for rec in self.index:
            grp = self._group_map[rec["group"]]
            self.versions.append(grp["frames"][rec["center"]]["version"])

        # Validate strides divisibility once
        H, W = self.img_size
        for spec in self.heatmap_specs:
            s = int(spec.stride)
            assert H % s == 0 and W % s == 0, "img_size must be divisible by all heatmap strides"

        # Gaussian mesh caches
        self._mesh_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.index[idx]
        grp = self._group_map[rec["group"]]
        i = rec["center"]
        frames = grp["frames"]
        # Gather T indices according to frame_stride
        indices = list(range(i - self.r * self.frame_stride, i + self.r * self.frame_stride + 1, self.frame_stride))

        # Load images and keypoints for all frames
        imgs_rgb: List[np.ndarray] = []
        kps_list: List[List[Tuple[float, float]]] = []
        versions: List[str] = []
        file_names: List[str] = []
        for k in indices:
            f = frames[k]
            # Resolve path using original_path if present, else file_name; avoid double-root join
            rel = str(f["meta"].get("original_path") or f["meta"].get("file_name", ""))
            rel_path = Path(rel)
            if not rel_path.is_absolute():
                parts = rel_path.parts
                if len(parts) > 0 and parts[0] == self.img_dir.name:
                    rel_path = Path(*parts[1:])
                img_path = _to_abs(str(self.img_dir / rel_path))
            else:
                img_path = str(rel_path)
            im_bgr = cv2.imread(img_path)
            if im_bgr is None:
                raise FileNotFoundError(img_path)
            imgs_rgb.append(cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB))
            kps: List[Tuple[float, float]] = []
            if (
                f["ann"] is not None
                and isinstance(f["ann"].get("keypoints"), (list, tuple))
                and len(f["ann"]["keypoints"]) >= 3
            ):
                cx, cy, _v = map(float, f["ann"]["keypoints"][0:3])
                kps.append((cx, cy))
            kps_list.append(kps)
            versions.append(f["version"])
            file_names.append(str(f["meta"].get("file_name")))

        # Apply identical augmentations via ReplayCompose
        out0 = self.transform(image=imgs_rgb[0], keypoints=kps_list[0])
        replay = out0.get("replay", None)
        images_t: List[torch.Tensor] = [out0["image"]]
        keypoints_t: List[List[Tuple[float, float]]] = [out0.get("keypoints", [])]
        for t in range(1, len(imgs_rgb)):
            if replay is not None:
                out = A.ReplayCompose.replay(replay, image=imgs_rgb[t], keypoints=kps_list[t])
            else:
                out = self.transform(image=imgs_rgb[t], keypoints=kps_list[t])
            images_t.append(out["image"])
            keypoints_t.append(out.get("keypoints", []))

        images_t = [img.float() / 255.0 if img.dtype == torch.uint8 else img for img in images_t]

        T = len(images_t)
        tgt_hmaps_per_scale: List[torch.Tensor] = []
        tgt_offs_per_scale: List[torch.Tensor] = []
        T_coords = torch.full((T, 2), -1.0, dtype=torch.float32)
        T_valid_list: List[float] = []

        for spec in self.heatmap_specs:
            hmaps = []
            offs = []
            for t in range(T):
                # Use only in-bounds keypoint after transform
                kps_in: List[Tuple[float, float]] = []
                if len(keypoints_t[t]) > 0:
                    x, y = keypoints_t[t][0]
                    H, W = self.img_size
                    if 0.0 <= x < W and 0.0 <= y < H:
                        kps_in = [(float(x), float(y))]
                        T_coords[t] = torch.tensor([x, y], dtype=torch.float32)
                        if spec is self.heatmap_specs[0]:
                            T_valid_list.append(1.0)
                    else:
                        if spec is self.heatmap_specs[0]:
                            T_valid_list.append(0.0)
                else:
                    if spec is self.heatmap_specs[0]:
                        T_valid_list.append(0.0)

                h, o = self._generate_targets_for_scale(kps_in, spec)
                hmaps.append(torch.from_numpy(h).float())  # [1,Hs,Ws]
                offs.append(torch.from_numpy(o).float())  # [2,Hs,Ws]
            tgt_hmaps_per_scale.append(torch.stack(hmaps, dim=0))  # [T,1,Hs,Ws]
            tgt_offs_per_scale.append(torch.stack(offs, dim=0))  # [T,2,Hs,Ws]
        T_valid = torch.tensor(T_valid_list, dtype=torch.float32)

        # Compose sample with shape depending on sequence_length
        if self.L == 1:
            image = images_t[0]  # [3,H,W]
            heatmaps = [t[0] for t in tgt_hmaps_per_scale]  # list of [1,Hs,Ws]
            offsets = [t[0] for t in tgt_offs_per_scale]  # list of [2,Hs,Ws]
            coord = T_coords[0]
            valid_mask = T_valid[0]
        else:
            image = torch.stack(images_t, dim=0)  # [T,3,H,W]
            heatmaps = tgt_hmaps_per_scale
            offsets = tgt_offs_per_scale
            coord = T_coords  # [T,2]
            valid_mask = T_valid  # [T]

        sample = {
            "image": image,
            "heatmaps": heatmaps,
            "offsets": offsets,
            "coord": coord,
            "valid_mask": valid_mask,
            "meta": {
                "group": grp["group"],
                "center_index": i,
                "frame_indices": indices,
                "versions": versions,
                "file_names": file_names,
                "sequence_length": self.L,
                "frame_stride": self.frame_stride,
            },
        }
        return sample

    def _generate_targets_for_scale(
        self, kps_xy: List[Tuple[float, float]], spec: HeatmapSpec
    ) -> Tuple[np.ndarray, np.ndarray]:
        stride = int(spec.stride)
        H, W = self.img_size
        assert H % stride == 0 and W % stride == 0, "img_size must be divisible by all strides"
        hH, hW = H // stride, W // stride

        heatmap = np.zeros((1, hH, hW), dtype=np.float32)
        offsets = np.zeros((2, hH, hW), dtype=np.float32)

        if len(kps_xy) == 0:
            return heatmap, offsets

        cx, cy = kps_xy[0]
        cx_h = cx / stride
        cy_h = cy / stride

        ix = int(np.clip(np.floor(cx_h), 0, hW - 1))
        iy = int(np.clip(np.floor(cy_h), 0, hH - 1))

        sigma = float(spec.sigma)
        xx, yy = self._get_mesh(hH, hW)
        dist2 = (xx - cx_h) ** 2 + (yy - cy_h) ** 2
        heatmap[0] = np.exp(-dist2 / (2 * sigma * sigma))

        fx = float(cx_h - ix)
        fy = float(cy_h - iy)
        offsets[0, iy, ix] = fx
        offsets[1, iy, ix] = fy

        return heatmap, offsets

    def _get_mesh(self, hH: int, hW: int) -> Tuple[np.ndarray, np.ndarray]:
        key = (hH, hW)
        if key not in self._mesh_cache:
            xs = np.arange(hW, dtype=np.float32)
            ys = np.arange(hH, dtype=np.float32)
            self._mesh_cache[key] = np.meshgrid(xs, ys)
        return self._mesh_cache[key]

    # Transform property to ensure identical augmentations via ReplayCompose
    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, t):
        if isinstance(t, A.ReplayCompose):
            self._transform = t
        elif isinstance(t, A.Compose):
            kp = A.KeypointParams(format="xy", remove_invisible=False)
            self._transform = A.ReplayCompose(t.transforms, keypoint_params=kp)
        else:
            base = _default_transforms(self.img_size)
            kp = A.KeypointParams(format="xy", remove_invisible=False)
            self._transform = A.ReplayCompose(base.transforms, keypoint_params=kp)


def build_heatmap_specs(strides: List[int], sigmas: List[float]) -> List[HeatmapSpec]:
    assert len(strides) == len(sigmas), "strides and sigmas must have same length"
    return [HeatmapSpec(s, sg) for s, sg in zip(strides, sigmas)]
