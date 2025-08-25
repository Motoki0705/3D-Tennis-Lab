# Step 3.1: Domain state management
from dataclasses import dataclass, field
from typing import List, Set, Optional, Dict, Any
import copy
from ..core.types import Keypoint
from ..core.types import FitResult


def init_kps(n=15) -> List[Keypoint]:
    """Initializes a list of n keypoints."""
    return [Keypoint() for _ in range(n)]


@dataclass
class DragState:
    """Represents the state of a mouse drag operation."""

    active: bool = False
    kp_index: Optional[int] = None
    cursor_xy: Optional[tuple] = None


@dataclass
class FrameState:
    """Encapsulates all state for a single frame."""

    kps: List[Keypoint] = field(default_factory=lambda: init_kps(15))
    placed: Set[int] = field(default_factory=set)
    skipped: Set[int] = field(default_factory=set)
    locked: Set[int] = field(default_factory=set)
    drag: DragState = field(default_factory=DragState)
    undo_stack: List[Dict[str, Any]] = field(default_factory=list)
    redo_stack: List[Dict[str, Any]] = field(default_factory=list)
    # Cache last fit result to avoid recomputation each UI tick
    last_fit: Optional[FitResult] = None


def advance_focus(kps: List[Keypoint], cur: int) -> int:
    """
    Finds the next keypoint index that is not yet entered.
    Cycles through all 15 points starting from the current one.
    """
    for step in range(1, 16):
        i = (cur + step) % 15
        if kps[i].v == 0:
            return i
    return cur  # Return current focus if all points are filled


def snapshot_frame_state(fs: FrameState, focus_idx: int) -> Dict[str, Any]:
    """Creates a deep snapshot of the frame's editable state."""
    return {
        "kps": copy.deepcopy(fs.kps),
        "placed": set(fs.placed),
        "skipped": set(fs.skipped),
        "locked": set(fs.locked),
        "focus_idx": int(focus_idx),
    }


def restore_frame_state(fs: FrameState, snap: Dict[str, Any]) -> int:
    """Restores the frame state from a snapshot. Returns restored focus index."""
    fs.kps = copy.deepcopy(snap["kps"])  # ensure independence
    fs.placed = set(snap["placed"]) if "placed" in snap else set()
    fs.skipped = set(snap["skipped"]) if "skipped" in snap else set()
    fs.locked = set(snap["locked"]) if "locked" in snap else set()
    fs.drag = DragState()  # reset transient drag state
    return int(snap.get("focus_idx", 0))


def reset_frame_state(fs: FrameState):
    """Resets the frame state to a pristine state (does not affect undo/redo stacks)."""
    fs.kps = init_kps(15)
    fs.placed.clear()
    fs.skipped.clear()
    fs.locked.clear()
    fs.drag = DragState()
