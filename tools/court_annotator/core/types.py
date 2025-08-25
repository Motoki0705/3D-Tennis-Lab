# Step 2.1: Core type definitions
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Keypoint:
    x: Optional[float] = None
    y: Optional[float] = None
    v: int = 0  # 0: not entered, 1: occluded, 2: visible
    locked: bool = False
    skip: bool = False
    source: str = "user"  # "user" or "auto"


@dataclass
class CourtSpec:
    names: List[str]
    skeleton: List[Tuple[int, int]]
    template_xy: List[Tuple[float, float]]


@dataclass
class FitResult:
    H: Optional[list] = None  # 3x3 homography matrix
    used: int = 0
    rmse: Optional[float] = None
    inlier_idx: List[int] = field(default_factory=list)
