import cv2
from typing import List, Dict, Any


def _keycode_to_name(code: int) -> str | None:
    """Convert cv2.waitKeyEx code to a normalized key name.
    Returns None if the code is not recognized.

    Note: We prioritize arrow detection before ASCII letters to avoid
    misinterpreting arrow keycodes (e.g., 81..84 on some platforms)
    as 'q/r/s/t'.
    """
    if code == -1:
        return None
    # Arrow keys (common codes and Qt/Windows specific)
    arrow_map = {
        81: "left",
        82: "up",
        83: "right",
        84: "down",  # some platforms
        2424832: "left",
        2490368: "up",
        2555904: "right",
        2621440: "down",  # Windows
        65361: "left",
        65362: "up",
        65363: "right",
        65364: "down",  # X11/Qt
    }
    if code in arrow_map:
        return arrow_map[code]
    # Enter/Return
    if code in (10, 13):
        return "enter"
    # Space
    if code == 32:
        return "space"
    # Letters
    if 97 <= code <= 122:
        return chr(code)  # a-z
    if 65 <= code <= 90:
        return chr(code + 32)  # A-Z -> a-z
    # Control codes (Ctrl+A..Z => 1..26)
    if 1 <= code <= 26:
        return f"ctrl+{chr(ord('a') + code - 1)}"
    return None


class InputAdapter:
    """
    Converts cv2 keyboard/mouse callbacks into raw, platform-neutral event dicts.
    Does NOT perform bind resolution. Use drain() to consume queued events.
    """

    def __init__(self):
        self._queue: List[Dict[str, Any]] = []

    def handle_key(self, code: int):
        name = _keycode_to_name(code)
        if name is not None:
            self._queue.append({"type": "key", "key": name})

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._queue.append({"type": "mouse", "event": "lbutton_down", "xy": (x, y)})
        elif event == cv2.EVENT_LBUTTONUP:
            self._queue.append({"type": "mouse", "event": "lbutton_up", "xy": (x, y)})
        elif event == cv2.EVENT_MOUSEMOVE:
            self._queue.append({"type": "mouse", "event": "mouse_move", "xy": (x, y)})

    def drain(self) -> List[Dict[str, Any]]:
        out = self._queue
        self._queue = []
        return out
