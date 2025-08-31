import json
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


class BallDataset(Dataset):
    """
    Ball-only dataset from COCO-style annotations.

    - Uses only annotations with category_id == 1 (ball).
    - Generates multiscale Gaussian heatmaps and offset targets per scale.
    - Supports negative frames (no ball) when configured.
    - Supports version mixing labels (e.g., v1/v2) used by the DataModule sampler.
    """

    def __init__(
        self,
        img_dir: Path,
        annotation_file: Path,
        img_size: Tuple[int, int],
        heatmap_specs: List[HeatmapSpec],
        negatives: str = "use",  # none|use|skip
        version_field: str = "view",
        transform: Optional[A.Compose] = None,
    ):
        super().__init__()
        self.img_dir = Path(img_dir)
        self.img_size = tuple(img_size)
        self.heatmap_specs = heatmap_specs
        self.negatives = negatives
        self.version_field = version_field
        self.transform = transform or _default_transforms(self.img_size)

        with open(_to_abs(str(annotation_file)), "r") as f:
            coco = json.load(f)

        # Build image_id -> meta
        self.images: Dict[int, Dict[str, Any]] = {img["id"]: img for img in coco["images"]}

        # Group ball annotations per image_id
        balls_per_image: Dict[int, List[Dict[str, Any]]] = {}
        for ann in coco["annotations"]:
            if int(ann.get("category_id", -1)) != 1:
                continue
            img_id = int(ann["image_id"])
            balls_per_image.setdefault(img_id, []).append(ann)

        # Create index as per-image entries with optional negatives
        self.index: List[Dict[str, Any]] = []
        for img_id, meta in self.images.items():
            anns = balls_per_image.get(img_id, [])
            if len(anns) == 0 and self.negatives == "skip":
                continue
            # If multiple balls exist, choose the first (dataset expectation: usually one)
            chosen_ann = anns[0] if len(anns) > 0 else None
            self.index.append({"image_id": img_id, "image_meta": meta, "ann": chosen_ann})

        # Precompute version label per index for sampling
        self.versions: List[str] = []
        for rec in self.index:
            v = self._extract_version(rec["image_meta"]) or "unknown"
            self.versions.append(v)

        # Cache for gaussian mesh grids per (hH,hW)
        self._mesh_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

    def _extract_version(self, image_meta: Dict[str, Any]) -> Optional[str]:
        # Prefer explicit field
        if self.version_field in image_meta:
            val = image_meta[self.version_field]
            if isinstance(val, str):
                return val
            if isinstance(val, (list, tuple)) and len(val) > 0:
                return str(val[0])

        # Fallback: try parsing from file_name or path
        file_name = str(image_meta.get("file_name", "")).lower()
        if "/v1/" in file_name or "_v1" in file_name or "-v1" in file_name:
            return "v1"
        if "/v2/" in file_name or "_v2" in file_name or "-v2" in file_name:
            return "v2"
        return None

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.index[idx]
        image_meta = rec["image_meta"]
        ann = rec["ann"]

        img_path = _to_abs(str(self.img_dir / image_meta["file_name"]))
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Build keypoints list (center of bbox) if positive
        kps: List[Tuple[float, float]] = []
        if ann is not None:
            cx, cy, _v = map(float, ann["keypoints"])  # COCO bbox
            kps.append((cx, cy))

        # Albumentations transform with keypoints
        transformed = self.transform(image=image, keypoints=kps)
        image_t: torch.Tensor = transformed["image"]
        kps_t: List[Tuple[float, float]] = transformed.get("keypoints", [])

        # Image normalization to float32 in [0,1] assumed in transform
        if image_t.dtype == torch.uint8:
            image_t = image_t.float() / 255.0

        # Generate per-scale targets
        targets_hmaps: List[torch.Tensor] = []
        targets_offsets: List[torch.Tensor] = []
        valid = 1.0 if len(kps_t) > 0 else 0.0
        valid_mask = torch.tensor(valid, dtype=torch.float32)

        for spec in self.heatmap_specs:
            hmap, offsets = self._generate_targets_for_scale(kps_t, spec)
            targets_hmaps.append(torch.from_numpy(hmap).float())  # [1,H,W]
            targets_offsets.append(torch.from_numpy(offsets).float())  # [2,H,W]

        sample = {
            "image": image_t,
            "heatmaps": targets_hmaps,  # low->high resolution order matches specs
            "offsets": targets_offsets,
            "valid_mask": valid_mask,
            # GT coordinate in input image pixels (after transforms)
            "coord": torch.tensor(kps_t[0], dtype=torch.float32) if len(kps_t) else torch.tensor([-1.0, -1.0]),
            "meta": {
                "image_id": rec["image_id"],
                "file_name": image_meta.get("file_name"),
                "version": self.versions[idx],
                "img_size": tuple(self.img_size),
                "heatmap_specs": [(s.stride, s.sigma) for s in self.heatmap_specs],
            },
        }
        return sample

    def _generate_targets_for_scale(
        self, kps_xy: List[Tuple[float, float]], spec: HeatmapSpec
    ) -> Tuple[np.ndarray, np.ndarray]:
        stride = spec.stride
        H, W = self.img_size
        hH, hW = H // stride, W // stride

        heatmap = np.zeros((1, hH, hW), dtype=np.float32)
        offsets = np.zeros((2, hH, hW), dtype=np.float32)

        if len(kps_xy) == 0:
            return heatmap, offsets

        cx, cy = kps_xy[0]
        cx_h = cx / stride
        cy_h = cy / stride

        # Integer cell location
        ix = int(np.clip(np.floor(cx_h), 0, hW - 1))
        iy = int(np.clip(np.floor(cy_h), 0, hH - 1))

        # Generate Gaussian heatmap centered at (cx_h, cy_h) in heatmap coordinates
        sigma = float(spec.sigma)
        # Use cached mesh
        xx, yy = self._get_mesh(hH, hW)
        dist2 = (xx - cx_h) ** 2 + (yy - cy_h) ** 2
        heatmap[0] = np.exp(-dist2 / (2 * sigma * sigma))

        # Offsets: fractional part relative to the integer cell index
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


def build_heatmap_specs(strides: List[int], sigmas: List[float]) -> List[HeatmapSpec]:
    assert len(strides) == len(sigmas), "strides and sigmas must have same length"
    return [HeatmapSpec(s, sg) for s, sg in zip(strides, sigmas)]
