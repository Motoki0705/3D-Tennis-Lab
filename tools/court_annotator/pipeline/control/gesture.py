from typing import List, Dict, Any, Optional
import numpy as np


def _find_nearby_point(pos, kps, radius) -> Optional[int]:
    for i, kp in enumerate(kps):
        if kp.v > 0 and kp.x is not None and kp.y is not None:
            if np.linalg.norm(np.array(pos) - np.array([kp.x, kp.y])) < radius:
                return i
    return None


class GestureRecognizer:
    """
    Converts raw mouse events into high-level gestures:
      - drag_start(idx), drag_move(xy), drag_end
      - place_at_focus(xy)
    Does not mutate state; only emits commands for ControlStage.
    """

    def __init__(self, snap_radius_px: int):
        self.snap_radius_px = int(snap_radius_px)

    def recognize(self, raw_events: List[Dict[str, Any]], frame_state) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for ev in raw_events:
            if ev.get("type") != "mouse":
                continue
            et = ev.get("event")
            xy = ev.get("xy")
            if frame_state.drag.active:
                if et == "mouse_move":
                    out.append({"cmd": "drag_move", "xy": xy})
                elif et == "lbutton_up":
                    out.append({"cmd": "drag_end"})
            else:
                if et == "lbutton_down":
                    nearby = _find_nearby_point(xy, frame_state.kps, self.snap_radius_px)
                    if nearby is not None:
                        out.append({"cmd": "drag_start", "idx": nearby})
                    else:
                        out.append({"cmd": "place_at_focus", "xy": xy})
        return out
