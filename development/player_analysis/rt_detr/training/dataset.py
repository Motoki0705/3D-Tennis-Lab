from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


@dataclass
class DataConfig:
    images_root: str
    labeled_json: str
    img_size: Tuple[int, int]
    category_name: str = "player"


def _safe_path(im: Dict[str, Any]) -> str:
    return im.get("original_path") or im.get("file_name") or ""


def _natural_key(s: str):
    import re

    return [int(t) if t.isdigit() else t for t in re.findall(r"\d+|\D+", s)]


class PlayerDetectionDataset(Dataset):
    """Single-image dataset for player detection using COCO-like annotations.

    - Expects COCO-style JSON with `images`, `annotations`, `categories`.
    - Filters annotations by `category_name` (default: 'player').
    - Returns (image_tensor, target_dict) for each image.
    - target_dict: {boxes(cxcywh_norm), labels, image_id, orig_size, size}
    """

    def __init__(
        self,
        cfg: DataConfig,
        transforms=None,
        *,
        allowed_image_ids: Optional[List[int]] = None,
        data_override: Optional[Dict[str, Any]] = None,
    ):
        self.cfg = cfg
        self.transforms = transforms
        self.data = data_override if data_override is not None else self._load_json(cfg.labeled_json)

        self.cat_id = self._get_category_id(self.data, cfg.category_name)
        self.images: List[Dict[str, Any]] = self.data.get("images", [])

        # Map image_id -> list of bboxes/labels for the requested category
        self.ann_map: Dict[int, List[Tuple[List[float], int]]] = {}
        for ann in self.data.get("annotations", []) or []:
            if int(ann.get("category_id", -1)) != self.cat_id:
                continue
            img_id = int(ann.get("image_id"))
            bbox = ann.get("bbox")  # COCO [x,y,w,h]
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            self.ann_map.setdefault(img_id, []).append((list(map(float, bbox)), 1))  # single class: 1

        if allowed_image_ids is not None:
            selected = set(int(i) for i in allowed_image_ids)
            self.images = [im for im in self.images if int(im.get("id")) in selected]

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return json.load(f)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        im_meta = self.images[idx]
        img_id = int(im_meta.get("id"))
        path = _safe_path(im_meta)
        path = path if os.path.isabs(path) else os.path.join(self.cfg.images_root, path)

        with Image.open(path) as im:
            image_np = np.array(im.convert("RGB"))
        H, W = image_np.shape[:2]

        # BBoxes in COCO format, with dummy label 1 (player)
        boxes = [b for (b, _) in self.ann_map.get(img_id, [])]
        labels = [1 for _ in self.ann_map.get(img_id, [])]

        if self.transforms is not None:
            transformed = self.transforms(image=image_np, bboxes=boxes, class_labels=labels)
            image_t = transformed["image"]  # tensor C,H,W
            boxes_t = transformed["bboxes"]
            labels_t = transformed["class_labels"]
        else:
            # As a fallback, convert to tensor without resize/normalize
            import torchvision.transforms.functional as TF

            image_t = TF.to_tensor(image_np)
            boxes_t = boxes
            labels_t = labels

        # Convert abs COCO xywh to normalized cxcywh
        _, out_h, out_w = image_t.shape
        boxes_cxcywh = []
        for x, y, w, h in boxes_t:
            cx = (x + w / 2.0) / float(out_w)
            cy = (y + h / 2.0) / float(out_h)
            boxes_cxcywh.append([cx, cy, w / float(out_w), h / float(out_h)])

        target = {
            "boxes": torch.tensor(boxes_cxcywh, dtype=torch.float32),
            "labels": torch.tensor(labels_t, dtype=torch.int64),
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "orig_size": torch.tensor([H, W], dtype=torch.int64),
            "size": torch.tensor([out_h, out_w], dtype=torch.int64),
        }

        return image_t, target

    @staticmethod
    def _get_category_id(data: Dict[str, Any], category_name: str) -> int:
        cats = data.get("categories", []) or []
        if not cats:
            # Default to 1 if categories are not present
            return 1
        for c in cats:
            if str(c.get("name", "")).lower() == str(category_name).lower():
                return int(c.get("id", 1))
        return int(cats[0].get("id", 1))


def split_by_clip_groups(data: Dict[str, Any], val_ratio: float, split_seed: int) -> Tuple[List[int], List[int]]:
    images = data.get("images", []) or []
    have_ids = all(("game_id" in im and "clip_id" in im) for im in images) and len(images) > 0
    if have_ids:
        groups: Dict[Tuple[int, int], List[int]] = {}
        for im in images:
            groups.setdefault((int(im.get("game_id")), int(im.get("clip_id"))), []).append(int(im.get("id")))
        group_keys = sorted(groups.keys())
        import random

        rng = random.Random(int(split_seed))
        rng.shuffle(group_keys)
        n_val = max(1, int(round(len(group_keys) * float(val_ratio)))) if len(group_keys) > 1 else 0
        val_g = set(group_keys[:n_val])
        train_ids, val_ids = [], []
        for k, ids in groups.items():
            (val_ids if k in val_g else train_ids).extend(ids)
        return sorted(train_ids), sorted(val_ids)
    else:
        # Fallback: group by parent directory to avoid leakage
        from collections import defaultdict

        groups = defaultdict(list)
        for im in images:
            groups[os.path.dirname(_safe_path(im))].append(int(im.get("id")))
        group_keys = sorted(groups.keys())
        import random

        rng = random.Random(int(split_seed))
        rng.shuffle(group_keys)
        n_val = max(1, int(round(len(group_keys) * float(val_ratio)))) if len(group_keys) > 1 else 0
        val_g = set(group_keys[:n_val])
        train_ids, val_ids = [], []
        for k, ids in groups.items():
            (val_ids if k in val_g else train_ids).extend(ids)
        return sorted(train_ids), sorted(val_ids)
