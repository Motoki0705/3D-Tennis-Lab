# Step 5.3: COCO format output utility
import json
from pathlib import Path
from typing import List, Dict
from ..core.types import Keypoint


def coco_keypoints_array(kps: List[Keypoint]):
    """Converts a list of Keypoint objects to a COCO-style flat array."""
    out = []
    for kp in kps:
        if kp.v == 0 or kp.x is None or kp.y is None:
            out.extend([0.0, 0.0, 0])
        else:
            out.extend([float(kp.x), float(kp.y), int(kp.v)])
    return out


def dump_coco(
    out_dir: Path,
    video_stem: str,
    images_meta: Dict,
    frames: Dict[int, List[Keypoint]],
    categories: List[Dict],
    fps: float,
    nframes: int,
    video_path: str,
):
    """Dumps annotations to ann.json and metadata to meta.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    for img_id, meta in sorted(images_meta.items()):
        images.append(dict(id=img_id, file_name=meta["file_name"], width=meta["width"], height=meta["height"]))
        annotations.append(
            dict(
                id=img_id,
                image_id=img_id,
                category_id=1,  # Assuming a single category
                keypoints=coco_keypoints_array(frames[img_id]),
                num_keypoints=sum(1 for kp in frames[img_id] if kp.v > 0),
                iscrowd=0,
            )
        )

    # Write annotations file
    ann_path = out_dir / "ann.json"
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(dict(images=images, annotations=annotations, categories=categories), f, ensure_ascii=False, indent=2)

    # Write metadata file
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(dict(video=video_path, fps=fps, nframes=nframes), f, ensure_ascii=False, indent=2)
