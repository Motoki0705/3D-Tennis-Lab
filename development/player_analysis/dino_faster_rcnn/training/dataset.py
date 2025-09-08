from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import cv2
import torch
from torch.utils.data import Dataset


class CocoDetDataset(Dataset):
    """Minimal COCO bbox dataset for Faster R-CNN training.

    Expects COCO-style instances JSON with bbox in [x,y,w,h] and category_id.
    """

    def __init__(self, images_dir: str, ann_file: str, transforms=None):
        self.images_dir = images_dir
        self.ann_file = ann_file
        self.transforms = transforms

        with open(ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.id_to_img = {img["id"]: img for img in coco.get("images", [])}
        self.img_to_anns: Dict[int, List[Dict[str, Any]]] = {img_id: [] for img_id in self.id_to_img.keys()}
        for ann in coco.get("annotations", []):
            if ann.get("iscrowd", 0) == 1:
                continue
            self.img_to_anns[ann["image_id"]].append(ann)
        self.img_ids = list(self.id_to_img.keys())

        # Optional mapping for category ids to contiguous labels 1..K
        cat_ids = sorted({ann["category_id"] for anns in self.img_to_anns.values() for ann in anns})
        self.catid_to_label = {cid: i + 1 for i, cid in enumerate(cat_ids)}  # 0 is background in torchvision

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        img_info = self.id_to_img[img_id]
        img_path = os.path.join(self.images_dir, img_info["file_name"])

        img = cv2.imread(img_path)
        assert img is not None, f"Failed to read image: {img_path}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        anns = self.img_to_anns.get(img_id, [])
        boxes = []
        labels = []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, w, h])
            labels.append(self.catid_to_label[a["category_id"]])

        class_labels = labels.copy()  # for albumentations

        if self.transforms is not None:
            transformed = self.transforms(image=img, bboxes=boxes, class_labels=class_labels)
            img_t = transformed["image"]
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
