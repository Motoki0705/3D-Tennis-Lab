from __future__ import annotations

from typing import Dict, Any
import numpy as np


def court3d_from_cfg(court_cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    court3d: Dict[str, np.ndarray] = {}
    if "keypoints" in court_cfg:
        for name, xyz in court_cfg.get("keypoints", {}).items():
            court3d[str(name)] = np.asarray(xyz, dtype=float).reshape(3)
        return court3d
    if "template_xy" in court_cfg and "names" in court_cfg:
        names = list(court_cfg["names"])  # ordered
        xy = np.asarray(court_cfg["template_xy"], dtype=float)
        length_m = float(court_cfg.get("dimensions", {}).get("length_m", 23.77))
        center_y = length_m / 2.0
        xy_centered = xy.copy()
        xy_centered[:, 1] = xy_centered[:, 1] - center_y
        for name, (x, y) in zip(names, xy_centered):
            court3d[str(name)] = np.array([float(x), float(y), 0.0], dtype=float)
        return court3d
    raise ValueError("court3d_from_cfg: unknown court config schema")
