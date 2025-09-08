from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable

import json
import os

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


@dataclass
class DataConfig:
    images_root: str
    ann_file: str
    img_size: Tuple[int, int]


class CocoDetectionDataset(Dataset):
    """Minimal COCO detection dataset without requiring pycocotools.

    Expects `ann_file` JSON with at least images/annotations/categories.
    Produces per-sample: image Tensor(C,H,W) in [0,1] (after transforms) and target dict:
      - boxes: Tensor(N,4) [x1,y1,x2,y2]
      - labels: Tensor(N)
      - image_id: Tensor(1)
    """

    def __init__(
        self,
        cfg: DataConfig,
        transforms: Optional[Callable] = None,
        *,
        keep_image_indices: Optional[List[int]] = None,
        preloaded: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.cfg = cfg
        self.transforms = transforms
        data = preloaded if preloaded is not None else self._load_json(cfg.ann_file)

        self.images: List[Dict[str, Any]] = data.get("images", [])
        self.annotations: List[Dict[str, Any]] = data.get("annotations", [])
        self.categories: List[Dict[str, Any]] = data.get("categories", [])

        # Map original category ids -> contiguous [0..K-1]
        cat_ids = [int(c.get("id")) for c in self.categories] if self.categories else []
        if cat_ids:
            cat_ids_sorted = sorted(set(cat_ids))
            self.cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids_sorted)}
        else:
            # Default single class when categories missing
            self.cat_id_to_idx = {}

        # map image_id -> anns
        self.anns_per_image: Dict[int, List[Dict[str, Any]]] = {}
        for ann in self.annotations:
            img_id = int(ann.get("image_id"))
            self.anns_per_image.setdefault(img_id, []).append(ann)

        # ordered image_ids matching images list
        self.image_ids: List[int] = [int(im.get("id")) for im in self.images]

        if keep_image_indices is not None:
            self.image_ids = [self.image_ids[i] for i in keep_image_indices if 0 <= i < len(self.image_ids)]
            self.images = [self.images[i] for i in keep_image_indices if 0 <= i < len(self.images)]

        self.preloaded = data if preloaded is None else preloaded

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return json.load(f)

    def __len__(self) -> int:
        return len(self.image_ids)

    def _img_path(self, im: Dict[str, Any]) -> str:
        file_name = im.get("file_name") or im.get("path") or im.get("original_path")
        p = os.path.join(self.cfg.images_root, file_name)
        return p

    def _load_image(self, im: Dict[str, Any]) -> np.ndarray:
        p = self._img_path(im)
        with Image.open(p) as img:
            return np.array(img.convert("RGB"))

    @staticmethod
    def _xywh_to_xyxy(box: List[float]) -> List[float]:
        x, y, w, h = box
        return [x, y, x + w, y + h]

    def _build_targets(self, im: Dict[str, Any], H: int, W: int) -> Dict[str, torch.Tensor]:
        img_id = int(im.get("id"))
        anns = self.anns_per_image.get(img_id, [])
        boxes_xyxy: List[List[float]] = []
        labels: List[int] = []
        for a in anns:
            bbox = a.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            bx = self._xywh_to_xyxy([float(v) for v in bbox])
            # sanity clip
            bx[0] = max(0.0, min(bx[0], W - 1))
            bx[1] = max(0.0, min(bx[1], H - 1))
            bx[2] = max(0.0, min(bx[2], W))
            bx[3] = max(0.0, min(bx[3], H))
            if bx[2] <= bx[0] or bx[3] <= bx[1]:
                continue
            boxes_xyxy.append(bx)
            raw_cid = int(a.get("category_id", 1))
            if self.cat_id_to_idx:
                labels.append(int(self.cat_id_to_idx.get(raw_cid, 0)))
            else:
                labels.append(raw_cid)
        boxes_t = torch.tensor(boxes_xyxy, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.int64)
        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
        }
        return target

    def __getitem__(self, idx: int):
        im = self.images[idx]
        img_np = self._load_image(im)
        H, W = img_np.shape[0], img_np.shape[1]
        target = self._build_targets(im, H, W)

        if self.transforms is not None:
            # Albumentations expects bboxes as list of [x_min, y_min, x_max, y_max]
            bboxes = target["boxes"].tolist() if target["boxes"].numel() > 0 else []
            labels = target["labels"].tolist() if target["labels"].numel() > 0 else []
            data = self.transforms(image=img_np, bboxes=bboxes, class_labels=labels)
            img_t = data["image"]
            b = torch.tensor(data["bboxes"], dtype=torch.float32)
            l = torch.tensor(data.get("class_labels", labels), dtype=torch.int64)
            target = {
                "boxes": b,
                "labels": l,
                "image_id": target["image_id"],
            }
        else:
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        return img_t, target
