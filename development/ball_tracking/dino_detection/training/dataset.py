from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


def _load_annotations(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    txt = path.read_text(encoding="utf-8")
    txt = txt.strip()
    if not txt:
        return []
    # Try JSON list first
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    # Fallback: JSON Lines
    items: List[dict] = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


@dataclass
class DatasetConfig:
    images_root: str
    labeled_json: str
    img_size: Tuple[int, int] = (640, 640)
    output_stride: int = 4
    sigma_px: float = 2.0


class BallHeatmapDataset(Dataset):
    """Image → 1ch heatmap dataset.

    Expects annotation entries with fields:
      - image: relative path from images_root (or absolute)
      - center: [x, y] in original pixel coords (optional for unlabeled)
    """

    def __init__(self, cfg: DatasetConfig) -> None:
        super().__init__()
        self.images_root = Path(cfg.images_root)
        self.items = _load_annotations(Path(cfg.labeled_json))
        self.img_size = tuple(cfg.img_size)
        self.out_stride = int(cfg.output_stride)
        self.sigma_px = float(cfg.sigma_px)

        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Resize(self.img_size, antialias=True),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _resolve_path(root: Path, p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else (root / q)

    @staticmethod
    def _gaussian_2d(h: int, w: int, cx: float, cy: float, sigma: float) -> np.ndarray:
        ys = np.arange(h, dtype=np.float32)
        xs = np.arange(w, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
        return g.astype(np.float32)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        img_path = self._resolve_path(self.images_root, item["image"]).resolve()
        img = Image.open(img_path).convert("RGB")

        orig_w, orig_h = img.size
        x = self.to_tensor(img)  # (3,H,W) resized
        H, W = x.shape[-2], x.shape[-1]
        h_out, w_out = H // self.out_stride, W // self.out_stride

        center = item.get("center")
        if center is None:
            # unlabeled -> return zeros heatmap
            y = torch.zeros((1, h_out, w_out), dtype=torch.float32)
        else:
            cx, cy = float(center[0]), float(center[1])
            # Map directly from original image coordinates to heatmap resolution
            sx = (w_out - 1) / max(1, orig_w - 1)
            sy = (h_out - 1) / max(1, orig_h - 1)
            cx_hm = cx * sx
            cy_hm = cy * sy
            g = self._gaussian_2d(h_out, w_out, cx_hm, cy_hm, sigma=self.sigma_px)
            y = torch.from_numpy(g).unsqueeze(0)

        return x, y


__all__ = ["DatasetConfig", "BallHeatmapDataset"]
