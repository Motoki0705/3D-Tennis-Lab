from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Dict

import numpy as np

from ..base import Bundle
from ...core.camera import Intrinsics
from ...io.reader import adapt_frames, load_camera_index


def _to_posix(p: str) -> str:
    # Store as POSIX even on Windows
    return str(PurePosixPath(p))


class LoadInputs:
    """Load court 3D, camera intrinsics, and frame observations from config.

    This minimal version expects the Hydra config to pass dictionaries for
    - court_cfg: { keypoints: {name: [X,Y,0]}, skeleton: [...] }
    - camera_cfg: { fx, fy, cx, cy, skew?, dist? }
    - data_cfg: either
        - { frames: [{ frame_idx, image_path, keypoints: {name: [u,v,vis,w]} }] }
        - or { annotations_path: "data/annotations/*.jsonl" }  # not implemented yet
    """

    def __init__(self, court_cfg: Dict[str, Any], camera_cfg: Dict[str, Any], data_cfg: Dict[str, Any]):
        self.court_cfg = court_cfg
        self.camera_cfg = camera_cfg
        self.data_cfg = data_cfg

    def run(self, B: Bundle) -> Bundle:
        court3d: Dict[str, np.ndarray] = {}
        # Support two schemas:
        # 1) Legacy: { keypoints: {name: [X,Y,0]} }
        # 2) Court annotator spec: { names: [...], template_xy: [[x,y],...], dimensions.length_m }
        if "keypoints" in self.court_cfg:
            for name, xyz in self.court_cfg.get("keypoints", {}).items():
                court3d[name] = np.asarray(xyz, dtype=float).reshape(3)
        elif "template_xy" in self.court_cfg and "names" in self.court_cfg:
            names = list(self.court_cfg["names"])  # ordered
            xy = np.asarray(self.court_cfg["template_xy"], dtype=float)
            # Shift origin from near baseline center (y=0) to court center (y=L/2)
            length_m = float(self.court_cfg.get("dimensions", {}).get("length_m", 23.77))
            center_y = length_m / 2.0
            xy_centered = xy.copy()
            xy_centered[:, 1] = xy_centered[:, 1] - center_y
            for name, (x, y) in zip(names, xy_centered):
                court3d[str(name)] = np.array([float(x), float(y), 0.0], dtype=float)
        else:
            raise ValueError("LoadInputs: court config schema not recognized (expected keypoints or template_xy/names)")

        # Frames via adapter
        frames, warnings = adapt_frames(self.data_cfg)

        # Camera intrinsics: single or by camera index mapping
        K = Intrinsics.from_dict(self.camera_cfg)
        cam_index_path = self.data_cfg.get("camera_index")
        if cam_index_path:
            mapping = load_camera_index(cam_index_path)
            # naive prefix match on basename
            import os

            if frames:
                bn = os.path.basename(frames[0].image_path)
                chosen = None
                for pref, intr_path in mapping.items():
                    if bn.startswith(pref):
                        chosen = intr_path
                        break
                if chosen:
                    from omegaconf import OmegaConf

                    cfg = OmegaConf.load(chosen)
                    K = Intrinsics.from_dict(dict(cfg))

        B.court3d = court3d
        B.K = {
            "fx": K.fx,
            "fy": K.fy,
            "cx": K.cx,
            "cy": K.cy,
            "skew": K.skew,
            "dist": {
                "k1": K.dist.k1,
                "k2": K.dist.k2,
                "p1": K.dist.p1,
                "p2": K.dist.p2,
                "k3": K.dist.k3,
            },
        }
        B.frames = frames
        B.report["loaded"] = {
            "n_keypoints": len(court3d),
            "n_frames": len(frames),
        }
        if warnings:
            B.report.setdefault("warnings", {})
            B.report["warnings"].setdefault("io", []).extend(warnings)
        return B
