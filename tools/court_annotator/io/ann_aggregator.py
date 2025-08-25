from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, List
from .resume_manager import alloc_image_id
from ..core.types import Keypoint
from .coco_writer import coco_keypoints_array


def _load_ann(ann_path: Path) -> Dict:
    if ann_path.exists():
        with open(ann_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"images": [], "annotations": [], "categories": []}


def _save_ann(ann_path: Path, data: Dict):
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_or_update(
    out_dir: Path,
    categories: List[Dict],
    images_meta: Dict[int, Dict],
    frames: Dict[int, List[Keypoint]],
    resume: Dict,
    video_path: str,
):
    ann_path = out_dir / "ann.json"
    data = _load_ann(ann_path)
    # Initialize categories if empty
    if not data.get("categories"):
        data["categories"] = categories

    # Build index maps for fast update
    img_idx = {img["id"]: i for i, img in enumerate(data.get("images", []))}
    ann_idx = {ann["image_id"]: i for i, ann in enumerate(data.get("annotations", []))}

    for frame_idx, meta in sorted(images_meta.items()):
        # Allocate global image id
        gid = alloc_image_id(resume, video_path, int(frame_idx))
        # Image entry
        img_entry = dict(id=gid, file_name=meta["file_name"], width=meta["width"], height=meta["height"])
        if gid in img_idx:
            data["images"][img_idx[gid]] = img_entry
        else:
            data.setdefault("images", []).append(img_entry)
            img_idx[gid] = len(data["images"]) - 1

        # Annotation entry (one per image)
        kps = frames[int(frame_idx)]
        ann_entry = dict(
            id=gid,
            image_id=gid,
            category_id=1,
            keypoints=coco_keypoints_array(kps),
            num_keypoints=sum(1 for kp in kps if kp.v > 0),
            iscrowd=0,
        )
        if gid in ann_idx:
            data["annotations"][ann_idx[gid]] = ann_entry
        else:
            data.setdefault("annotations", []).append(ann_entry)
            ann_idx[gid] = len(data["annotations"]) - 1

    _save_ann(ann_path, data)
