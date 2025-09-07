from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any

import numpy as np


@dataclass
class Distortion:
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any] | None) -> "Distortion":
        if not d:
            return cls()
        return cls(
            k1=float(d.get("k1", 0.0)),
            k2=float(d.get("k2", 0.0)),
            p1=float(d.get("p1", 0.0)),
            p2=float(d.get("p2", 0.0)),
            k3=float(d.get("k3", 0.0)),
        )


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    skew: float = 0.0
    dist: Distortion = field(default_factory=Distortion)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Intrinsics":
        dist = Distortion.from_dict(d.get("dist"))
        return cls(
            fx=float(d.get("fx", 0.0)),
            fy=float(d.get("fy", 0.0)),
            cx=float(d.get("cx", 0.0)),
            cy=float(d.get("cy", 0.0)),
            skew=float(d.get("skew", 0.0) or 0.0),
            dist=dist,
        )


@dataclass
class PoseCW:
    """World -> Camera pose."""

    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)
