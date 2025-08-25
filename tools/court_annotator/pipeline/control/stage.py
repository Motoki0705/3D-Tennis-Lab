from ..runner import Stage
from .keymap import KeyMapper
from .gesture import GestureRecognizer
from ...domain.state import (
    advance_focus,
    snapshot_frame_state,
    restore_frame_state,
    reset_frame_state,
)


class ControlStage(Stage):
    name = "control"

    def __init__(self):
        self._keymapper = None

    def _ensure_keymapper(self, cfg):
        binds = cfg.get("ui", {}).get("binds", {})
        # Normalize a few aliases
        norm = {}
        for cmd, keys in binds.items():
            c = cmd.lower()
            if c == "save":
                c = "save_annotations"
            if c == "reset":
                c = "reset_frame"
            norm[c] = keys
        self._keymapper = KeyMapper(norm)

    def _push_undo(self, frame_state, focus_idx):
        frame_state.undo_stack.append(snapshot_frame_state(frame_state, focus_idx))
        frame_state.redo_stack.clear()

    def _apply_key_command(self, cmd_type, bundle):
        frame_state = bundle["frame_state"]
        focus_idx = bundle["focus_idx"]
        cfg = bundle.get("config", {})
        session_state = bundle.get("session_state", {})

        if cmd_type == "save_annotations":
            bundle["should_export"] = True
            bundle["hud_message"] = "Annotations saved!"
            return
        if cmd_type == "quit":
            bundle["should_export"] = True
            bundle["request_quit"] = True
            bundle["hud_message"] = "Saved and exiting..."
            return
        if cmd_type in ("mark_done", "done", "done_video"):
            # Mark current video done and move to next
            bundle["should_export"] = True
            bundle["request_done"] = True
            bundle["hud_message"] = "Marked video as done"
            return
        if cmd_type == "undo":
            if frame_state.undo_stack:
                frame_state.redo_stack.append(snapshot_frame_state(frame_state, focus_idx))
                snap = frame_state.undo_stack.pop()
                bundle["focus_idx"] = restore_frame_state(frame_state, snap)
                bundle["hud_message"] = "Undo"
                bundle["dirty"] = True
            else:
                bundle["hud_message"] = "Nothing to undo"
            return
        if cmd_type == "redo":
            if frame_state.redo_stack:
                frame_state.undo_stack.append(snapshot_frame_state(frame_state, focus_idx))
                snap = frame_state.redo_stack.pop()
                bundle["focus_idx"] = restore_frame_state(frame_state, snap)
                bundle["hud_message"] = "Redo"
                bundle["dirty"] = True
            else:
                bundle["hud_message"] = "Nothing to redo"
            return
        if cmd_type == "reset_frame":
            self._push_undo(frame_state, focus_idx)
            reset_frame_state(frame_state)
            bundle["focus_idx"] = 0
            bundle["hud_message"] = "Frame reset"
            bundle["dirty"] = True
            return
        if cmd_type == "mark_skip":
            self._push_undo(frame_state, focus_idx)
            kp = frame_state.kps[focus_idx]
            kp.skip = not kp.skip
            if kp.skip:
                frame_state.skipped.add(focus_idx)
                kp.x = kp.y = None
                kp.v = 0
                kp.source = "user"
            else:
                frame_state.skipped.discard(focus_idx)
            bundle["focus_idx"] = advance_focus(frame_state.kps, focus_idx)
            bundle["dirty"] = True
            return
        if cmd_type == "toggle_lock":
            self._push_undo(frame_state, focus_idx)
            kp = frame_state.kps[focus_idx]
            if kp.v > 0:
                kp.locked = not kp.locked
                if kp.locked:
                    frame_state.locked.add(focus_idx)
                else:
                    frame_state.locked.discard(focus_idx)
            bundle["dirty"] = True
            return
        if cmd_type == "toggle_visibility":
            self._push_undo(frame_state, focus_idx)
            kp = frame_state.kps[focus_idx]
            if kp.v > 0 and kp.source == "user":
                kp.v = 1 if kp.v == 2 else 2
            bundle["dirty"] = True
            return
        if cmd_type == "focus_next":
            bundle["focus_idx"] = (focus_idx + 1) % 15
            return
        if cmd_type == "focus_prev":
            bundle["focus_idx"] = (focus_idx - 1 + 15) % 15
            return
        if cmd_type == "recompute":
            bundle["hud_message"] = "Recomputed"
            bundle["force_recompute"] = True
            return
        if cmd_type == "copy_prev":
            # Copy from the immediate previous frame by stride, only into empty slots.
            self._push_undo(frame_state, focus_idx)
            stride = cfg.get("stride", 1)
            prev_idx = bundle.get("frame_idx", 0) - stride
            prev_fs = session_state.get("frame_states", {}).get(prev_idx)
            if prev_fs is not None:
                any_filled = any(kp.v > 0 for kp in frame_state.kps)
                changed = False
                for i in range(len(frame_state.kps)):
                    cur_kp = frame_state.kps[i]
                    prev_kp = prev_fs.kps[i]
                    if cur_kp.v == 0:
                        if prev_kp.skip:
                            cur_kp.skip = True
                            cur_kp.x = cur_kp.y = None
                            cur_kp.v = 0
                            cur_kp.source = "user"
                            frame_state.skipped.add(i)
                            changed = True
                        elif prev_kp.v > 0 and prev_kp.x is not None and prev_kp.y is not None:
                            cur_kp.x, cur_kp.y = prev_kp.x, prev_kp.y
                            cur_kp.v = prev_kp.v
                            cur_kp.source = prev_kp.source
                            frame_state.placed.add(i)
                            frame_state.skipped.discard(i)
                            changed = True
                        if prev_kp.locked:
                            cur_kp.locked = True
                            frame_state.locked.add(i)
                if not any_filled and changed:
                    bundle["should_persist_frame"] = True
                if changed:
                    bundle["dirty"] = True
                bundle["hud_message"] = "Copied from previous frame"
            else:
                bundle["hud_message"] = "No previous frame to copy"
            return
        if cmd_type == "next_frame":
            step = int(cfg.get("nav", {}).get("arrow_step", cfg.get("stride", 1)))
            bundle["nav_delta"] = bundle.get("nav_delta", 0) + step
            return
        if cmd_type == "prev_frame":
            step = int(cfg.get("nav", {}).get("arrow_step", cfg.get("stride", 1)))
            bundle["nav_delta"] = bundle.get("nav_delta", 0) - step
            return
        if cmd_type == "confirm":
            # Defer decision to UI loop after ValidateStage populated is_ready
            bundle["request_confirm"] = True
            return

    def _apply_mouse_command(self, mcmd, bundle):
        frame_state = bundle["frame_state"]
        focus_idx = bundle["focus_idx"]
        if mcmd["cmd"] == "drag_start":
            # snapshot once at drag start
            self._push_undo(frame_state, focus_idx)
            frame_state.drag.active = True
            frame_state.drag.kp_index = int(mcmd["idx"])
            bundle["focus_idx"] = int(mcmd["idx"])
        elif mcmd["cmd"] == "drag_move":
            if frame_state.drag.active and frame_state.drag.kp_index is not None:
                i = frame_state.drag.kp_index
                frame_state.kps[i].x, frame_state.kps[i].y = mcmd["xy"]
        elif mcmd["cmd"] == "drag_end":
            i = frame_state.drag.kp_index
            frame_state.drag.active = False
            frame_state.drag.kp_index = None
            # If a previously auto-completed point was dragged, promote it to user-specified
            if i is not None:
                kp = frame_state.kps[i]
                if kp.v > 0:
                    if not frame_state.placed:
                        bundle["should_persist_frame"] = True
                    kp.source = "user"
                    frame_state.placed.add(i)
                    frame_state.skipped.discard(i)
            bundle["dirty"] = True
        elif mcmd["cmd"] == "place_at_focus":
            # first placement => persist
            if not frame_state.placed:
                bundle["should_persist_frame"] = True
            self._push_undo(frame_state, focus_idx)
            kp = frame_state.kps[focus_idx]
            kp.x, kp.y = mcmd["xy"]
            kp.v = 2
            kp.source = "user"
            frame_state.placed.add(focus_idx)
            frame_state.skipped.discard(focus_idx)
            bundle["focus_idx"] = advance_focus(frame_state.kps, focus_idx)
            bundle["dirty"] = True

    def process(self, bundle):
        raw = bundle.get("raw_events")
        if not raw:
            return bundle

        if self._keymapper is None:
            self._ensure_keymapper(bundle.get("config", {}))

        # Map keys using binds from cfg
        key_cmds = self._keymapper.map_keys(raw)
        # Recognize mouse gestures
        snap_radius = bundle.get("config", {}).get("ui", {}).get("mouse", {}).get("snap_radius_px", 12)
        gestures = GestureRecognizer(snap_radius).recognize(raw, bundle["frame_state"])

        # Clear transient HUD
        bundle.pop("hud_message", None)

        # Apply key commands
        for kc in key_cmds:
            self._apply_key_command(kc["cmd"], bundle)

        # Apply mouse commands
        for mc in gestures:
            self._apply_mouse_command(mc, bundle)

        return bundle
