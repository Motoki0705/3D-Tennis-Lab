from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import cv2
import torch
from torch.utils.data import Dataset


class CocoDetDataset(Dataset):
    """COCO bbox dataset for Faster R-CNN training focused on players.

    - Expects COCO-style JSON (bbox in [x,y,w,h]).
    - Filters to the `player` category and keeps images with exactly two players.
    - Maps category ids to contiguous labels (player -> 1).
    """

    def __init__(
        self,
        images_dir: str,
        ann_file: str,
        transforms=None,
        target_category: str = "player",
        required_instances_per_image: int = 2,
    ):
        self.images_dir = images_dir
        self.ann_file = ann_file
        self.transforms = transforms
        self.target_category = target_category
        self.required_instances_per_image = required_instances_per_image

        with open(ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # Resolve target category id(s)
        cat_name_to_id = {c.get("name"): c.get("id") for c in coco.get("categories", [])}
        if target_category not in cat_name_to_id:
            raise ValueError(f"Category '{target_category}' not found in categories: {list(cat_name_to_id.keys())}")
        self.target_cat_id = cat_name_to_id[target_category]

        # Index images
        self.id_to_img = {img["id"]: img for img in coco.get("images", [])}

        # Collect only target-category annotations per image
        img_to_all_anns: Dict[int, List[Dict[str, Any]]] = {img_id: [] for img_id in self.id_to_img.keys()}
        for ann in coco.get("annotations", []):
            if ann.get("iscrowd", 0) == 1:
                continue
            if ann.get("category_id") != self.target_cat_id:
                continue
            # Basic bbox sanity check to avoid degenerate boxes
            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
            if w is None or h is None or w <= 1 or h <= 1:
                continue
            img_to_all_anns[ann["image_id"]].append(ann)

        # Keep only images with exactly the required number of target instances
        self.img_to_anns: Dict[int, List[Dict[str, Any]]] = {}
        for img_id, anns in img_to_all_anns.items():
            if len(anns) == self.required_instances_per_image:
                self.img_to_anns[img_id] = anns

        self.img_ids = list(self.img_to_anns.keys())

        # Mapping for category ids to contiguous labels 1..K (only target -> 1)
        self.catid_to_label = {self.target_cat_id: 1}  # 0 is background in torchvision

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        img_info = self.id_to_img[img_id]
        img_path = os.path.join(self.images_dir, img_info["original_path"])

        img = cv2.imread(img_path)
        assert img is not None, f"Failed to read image: {img_path}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        anns = self.img_to_anns.get(img_id, [])
        boxes = []
        labels = []
        H, W = img.shape[:2]
        for a in anns:
            x, y, w, h = a["bbox"]
            # Clip bbox to image bounds to satisfy albumentations' pre-check
            x1 = max(0.0, float(x))
            y1 = max(0.0, float(y))
            x2 = min(float(x + w), float(W))
            y2 = min(float(y + h), float(H))
            new_w = max(0.0, x2 - x1)
            new_h = max(0.0, y2 - y1)
            if new_w <= 1 or new_h <= 1:
                continue
            boxes.append([x1, y1, new_w, new_h])
            labels.append(self.catid_to_label[self.target_cat_id])

        class_labels = labels.copy()  # for albumentations

        if self.transforms is not None:
            transformed = self.transforms(image=img, bboxes=boxes, class_labels=class_labels)
            img_t = transformed["image"]
            # Safety: torchvision detectors expect float tensors in [0, 1]
            if img_t.dtype != torch.float32:
                img_t = img_t.float()
            # If values look like 0..255, rescale to 0..1
            if torch.isfinite(img_t).all() and img_t.max() > 1.0:
                img_t = img_t / 255.0
            boxes = transformed["bboxes"]
            labels = transformed["class_labels"]
        else:
            import torchvision.transforms.functional as F

            img_t = F.to_tensor(img)

        # Convert boxes from coco [x,y,w,h] to xyxy as expected by torchvision
        boxes_xyxy = []
        for x, y, w, h in boxes:
            boxes_xyxy.append([x, y, x + w, y + h])

        target: Dict[str, Any] = {
            "boxes": torch.tensor(boxes_xyxy, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            # optional fields (areas, iscrowd) can be added if needed
        }

        return img_t, target


def detection_collate(batch):
    images, targets = zip(*batch)
    return {"images": list(images), "targets": list(targets)}
