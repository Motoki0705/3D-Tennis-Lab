from __future__ import annotations

import json
import os
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import OmegaConf

from ..pipeline.base import FrameObs


def _to_posix(p: str) -> str:
    return str(PurePosixPath(str(p)))


def load_jsonl_frames(path: str, warnings: Optional[List[str]] = None) -> List[FrameObs]:
    frames: List[FrameObs] = []
    warns: List[str] = warnings if warnings is not None else []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            kps_in = rec.get("keypoints", {})
            kps = {}
            for k, v in kps_in.items():
                if isinstance(v, (list, tuple)):
                    if len(v) == 4:
                        u, v2, vis, w = v
                    elif len(v) == 3:
                        u, v2, vis = v
                        w = 1.0 if vis else 0.0
                    else:
                        u, v2 = v[:2]
                        vis, w = 1, 1.0
                else:
                    warns.append(f"JSONL: keypoint '{k}' not list/tuple; skipped")
                    continue
                kps[k] = (float(u), float(v2), int(vis), float(w))
            meta = rec.get("meta", {}) if isinstance(rec.get("meta", {}), dict) else {}
            frames.append(
                FrameObs(
                    frame_idx=int(rec.get("frame_idx", len(frames))),
                    image_path=_to_posix(str(rec.get("image_path", ""))),
                    keypoints=kps,
                    meta=meta,
                )
            )
    return frames


def load_camera_index(path: str) -> Dict[str, str]:
    cfg = OmegaConf.load(path)
    # Expect simple mapping: prefix -> intrinsics_yaml
    if isinstance(cfg, dict):
        return {str(k): str(v) for k, v in cfg.items()}
    else:
        # OmegaConf DictConfig
        return {str(k): str(v) for k, v in cfg.items()}


def load_coco_frames(
    path: str,
    category_name: Optional[str] = None,
    image_root: Optional[str] = None,
    selection: str = "best",  # "best" | "first"
    min_visible: int = 0,
    weight_policy: str = "fixed",  # "fixed" | "score"
    keypoint_map: Optional[Dict[str, str]] = None,
    warnings: Optional[List[str]] = None,
) -> List[FrameObs]:
    """Load frames from a COCO-style keypoints JSON.

    - Assumes the target category defines `keypoints: [name,...]` and annotations have
      `keypoints: [x1,y1,v1, x2,y2,v2, ...]` (COCO visibility v in {0,1,2}).
    - If multiple annotations per image exist for the category, picks the one with the
      most visible keypoints (ties broken by larger `score` if present).
    - `image_root` is optionally prefixed to the image file_name.
    """
    warns: List[str] = warnings if warnings is not None else []
    with open(path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco.get("images", [])}
    categories = {cat["id"]: cat for cat in coco.get("categories", [])}
    # pick category
    cat_id = None
    if category_name is not None:
        for cid, cat in categories.items():
            if str(cat.get("name")) == str(category_name):
                cat_id = cid
                break
    if cat_id is None:
        # fallback: first category that has keypoints defined
        for cid, cat in categories.items():
            if cat.get("keypoints"):
                cat_id = cid
                break
    if cat_id is None:
        raise ValueError("COCO: no category with keypoints found; specify data.coco_category")

    kp_names = categories[cat_id].get("keypoints", [])
    _name_count = len(kp_names)

    # group annotations per image
    ann_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco.get("annotations", []):
        if ann.get("category_id") != cat_id:
            continue
        img_id = ann.get("image_id")
        if img_id not in ann_by_img:
            ann_by_img[img_id] = []
        ann_by_img[img_id].append(ann)

    frames: List[FrameObs] = []
    for img_id, img in images.items():
        file_name = img.get("file_name", "")
        if image_root:
            image_path = _to_posix(os.path.join(image_root, file_name))
        else:
            image_path = _to_posix(file_name)

        anns = ann_by_img.get(img_id, [])
        if not anns:
            # No annotation; still create an empty keypoints frame
            meta = {"width": img.get("width"), "height": img.get("height"), "image_id": img.get("id")}
            frames.append(
                FrameObs(frame_idx=int(img.get("id", len(frames))), image_path=image_path, keypoints={}, meta=meta)
            )
            continue

        # choose best annotation
        if selection == "first":
            best = anns[0]
        else:

            def ann_score(a: Dict[str, Any]) -> tuple:
                kps = a.get("keypoints", [])
                vis_count = sum(1 for i in range(2, len(kps), 3) if kps[i] and kps[i] > 0)
                scr = float(a.get("score", 0.0))
                return (vis_count, scr)

            best = max(anns, key=ann_score)
        kplist = best.get("keypoints", [])
        ann_score_val = float(best.get("score", 1.0))
        kps: Dict[str, tuple[float, float, int, float]] = {}
        for idx, name in enumerate(kp_names):
            j = 3 * idx
            if j + 2 >= len(kplist):
                continue
            u = float(kplist[j])
            v = float(kplist[j + 1])
            vis_flag = int(kplist[j + 2])
            vis = 1 if vis_flag > 0 else 0
            if weight_policy == "score":
                w = float(ann_score_val if vis else 0.0)
            else:
                w = 1.0 if vis else 0.0
            nm = str(name)
            if keypoint_map and nm in keypoint_map:
                nm = str(keypoint_map[nm])
            kps[nm] = (u, v, vis, w)
        vis_cnt = sum(1 for _, (_, _, vv, ww) in kps.items() if vv and ww > 0)
        if vis_cnt < int(min_visible):
            warns.append(
                f"COCO: image_id={img_id} has only {vis_cnt} visible (< min_visible={min_visible}); frame kept but may fail later"
            )
        meta = {"width": img.get("width"), "height": img.get("height"), "image_id": img.get("id")}
        frames.append(
            FrameObs(frame_idx=int(img.get("id", len(frames))), image_path=image_path, keypoints=kps, meta=meta)
        )

    # sort by frame_idx for determinism
    frames.sort(key=lambda fr: fr.frame_idx)
    return frames


def adapt_frames(data_cfg: Dict[str, Any]) -> Tuple[List[FrameObs], List[str]]:
    """Adapter entry: normalize various input formats into FrameObs list + warnings.

    Supported configs:
      - { jsonl: path }
      - { coco: path, coco_category: name, image_root: dir, selection, min_visible, weight_policy, keypoint_map }
      - { frames: [...] }  # already normalized inline
    """
    warnings: List[str] = []
    if "jsonl" in data_cfg:
        frames = load_jsonl_frames(data_cfg["jsonl"], warnings=warnings)
        return frames, warnings
    if "coco" in data_cfg:
        frames = load_coco_frames(
            data_cfg["coco"],
            category_name=data_cfg.get("coco_category"),
            image_root=data_cfg.get("image_root"),
            selection=str(data_cfg.get("selection", "best")),
            min_visible=int(data_cfg.get("min_visible", 0) or 0),
            weight_policy=str(data_cfg.get("weight_policy", "fixed")),
            keypoint_map=data_cfg.get("keypoint_map"),
            warnings=warnings,
        )
        return frames, warnings
    if "frames" in data_cfg:
        frs: List[FrameObs] = []
        for f in data_cfg.get("frames", []):
            kps_in = f.get("keypoints", {})
            kps = {}
            for k, v in kps_in.items():
                if isinstance(v, (list, tuple)):
                    if len(v) == 4:
                        u, v2, vis, w = v
                    elif len(v) == 3:
                        u, v2, vis = v
                        w = 1.0 if vis else 0.0
                    else:
                        u, v2 = v[:2]
                        vis, w = 1, 1.0
                else:
                    warnings.append(f"frames: keypoint '{k}' not list/tuple; skipped")
                    continue
                kps[k] = (float(u), float(v2), int(vis), float(w))
            meta = f.get("meta", {}) if isinstance(f.get("meta", {}), dict) else {}
            frs.append(
                FrameObs(
                    frame_idx=int(f.get("frame_idx", len(frs))),
                    image_path=_to_posix(str(f.get("image_path", ""))),
                    keypoints=kps,
                    meta=meta,
                )
            )
        return frs, warnings
    raise RuntimeError("data config must specify one of: jsonl | coco | frames")
