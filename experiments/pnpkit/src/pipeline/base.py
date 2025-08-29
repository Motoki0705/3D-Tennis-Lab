from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Type

import numpy as np


@dataclass
class FrameObs:
    frame_idx: int
    image_path: str
    # keypoints: name -> (u, v, vis, weight)
    keypoints: Dict[str, tuple[float, float, int, float]]
    # optional metadata: width/height/camera_id/scene_id/etc.
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Bundle:
    bundle_id: Optional[str] = None
    court3d: Dict[str, np.ndarray] = None
    K: Optional[Dict[str, Any]] = None
    pose_cw: Optional[Dict[str, Any]] = None
    frames: List[FrameObs] = field(default_factory=list)
    poses: List[Dict[str, Any]] = field(default_factory=list)  # per-frame poses
    matches: Optional[List[tuple[str, str]]] = None
    H: Optional[np.ndarray] = None
    report: Dict[str, Any] = field(default_factory=dict)


class Stage:
    """Base class for pipeline stages with contract enforcement.

    Each stage may declare:
      - required_inputs: list of logical inputs (e.g., "frames", "court3d", "K", "pose_cw", "min_visible_4")
      - produces: list of outputs expected after run (e.g., "H", "pose_cw", "report.homography")
      - STAGE_VERSION: semantic version string, e.g., "1.0.0"
      - STAGE_NAME: set via @register or defaults to class name
    """

    required_inputs: List[str] = []
    produces: List[str] = []
    STAGE_VERSION: str = "1.0.0"
    STAGE_NAME: str = ""

    def __init__(self, **cfg):
        self.cfg = cfg

    def __call__(self, B: Bundle) -> Bundle:
        # Lazy import to avoid cycles
        from ..spec.contract import (
            ensure_versions,
            validate_required_inputs,
            validate_produces,
            add_stage_version,
        )

        # Ensure version stamps exist at the beginning of a run
        ensure_versions(B)
        # Precondition check
        validate_required_inputs(self, B)
        # Execute
        B = self.run(B)
        # Postcondition check
        validate_produces(self, B)
        # Record stage version for traceability
        add_stage_version(self, B)
        return B

    def run(self, B: Bundle) -> Bundle:  # pragma: no cover - interface
        raise NotImplementedError

    # DAG engine may consult this to auto-skip stages based on bundle state
    def should_skip(self, B: Bundle) -> bool:  # pragma: no cover - interface
        return False


REGISTRY: Dict[str, Type[Stage]] = {}


def register(name: str) -> Callable[[Type[Stage]], Type[Stage]]:
    def _wrap(cls: Type[Stage]) -> Type[Stage]:
        # Attach a stable stage name for contract/versions
        try:
            setattr(cls, "STAGE_NAME", name)
        except Exception:
            pass
        REGISTRY[name] = cls
        return cls

    return _wrap
