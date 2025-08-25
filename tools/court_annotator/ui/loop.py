# Step 7.3: Main UI Loop
import cv2
from .input_adapter import InputAdapter

WINDOW_NAME = "Court Annotator"


def run_ui_loop(cfg, runner, court_spec, video, video_id: int | None = None, resume_state: dict | None = None):
    """The main loop for the interactive UI."""
    cv2.namedWindow(WINDOW_NAME)
    adapter = InputAdapter()
    cv2.setMouseCallback(WINDOW_NAME, adapter.mouse_callback)

    frame_idx = cfg.get("start", 0)
    initial_focus_idx = cfg.get("init_focus_idx", 0)  # Initial focus
    focus_idx = initial_focus_idx

    # Session state to hold all annotation data and metadata
    session_state = {
        "frame_states": {},
        "saved_frames": set(),
        "images_meta": {},
    }

    scale = float(cfg.get("preprocess", {}).get("display_scale", 1.0) or 1.0)

    prev_frame_idx = None
    cached_frame = None

    stopped_by_quit = False
    stopped_by_done = False
    while True:
        # Detect frame change and avoid repeated random seeks
        frame_changed = frame_idx != prev_frame_idx
        if frame_changed or cached_frame is None:
            frame = video.read_at(frame_idx)
            cached_frame = frame
            prev_frame_idx = frame_idx
        else:
            frame = cached_frame
        if frame is None:
            print("End of video or failed to read frame.")
            break

        # --- Input ---
        key = cv2.waitKeyEx(1)
        if key != -1:
            adapter.handle_key(key)
        raw_events = adapter.drain()
        # Map mouse coordinates from display space back to original frame space
        if scale != 1.0 and raw_events:
            for ev in raw_events:
                if ev.get("type") == "mouse" and "xy" in ev:
                    x, y = ev["xy"]
                    ev["xy"] = (int(round(x / scale)), int(round(y / scale)))

        # --- Pipeline Execution ---
        bundle = {
            "frame_idx": frame_idx,
            "nframes": video.nframes,
            "frame": frame,
            "config": cfg,
            "court_spec": court_spec,
            "focus_idx": focus_idx,
            "session_state": session_state,
            "raw_events": raw_events,
            "video_path": video.path,  # Pass video path for metadata
            "video": video,  # Pass video object for source stage (fps etc.)
            "frame_changed": frame_changed,
            "video_id": video_id,
            "resume_state": resume_state,
        }
        bundle = runner.run_once(bundle)
        focus_idx = bundle["focus_idx"]  # Update focus from pipeline

        # --- Display ---
        rendered_frame = bundle.get("rendered_frame", frame)
        if scale != 1.0:
            disp = cv2.resize(
                rendered_frame, (int(rendered_frame.shape[1] * scale), int(rendered_frame.shape[0] * scale))
            )
        else:
            disp = rendered_frame
        cv2.imshow(WINDOW_NAME, disp)

        if bundle.get("request_quit"):
            stopped_by_quit = True
            break
        if bundle.get("request_done"):
            stopped_by_done = True
            break

        # --- Frame Navigation (handle in loop after command processing) ---
        nav_delta = int(bundle.get("nav_delta", 0))
        if nav_delta != 0:
            # Apply navigation and reset focus for the new frame
            frame_idx = max(0, frame_idx + nav_delta)
            focus_idx = initial_focus_idx
            bundle["nav_delta"] = 0
        if bundle.get("request_confirm"):
            if bundle.get("is_ready", False):
                confirm_step = cfg.get("nav", {}).get("confirm_step", cfg.get("stride", 1))
                frame_idx += int(confirm_step)
                # Reset focus on confirm-advance as we move to a new frame
                focus_idx = initial_focus_idx
            bundle["request_confirm"] = False

    cv2.destroyAllWindows()
    print("UI loop finished.")
    return {
        "last_frame_idx": frame_idx,
        "stopped_by_quit": stopped_by_quit,
        "stopped_by_done": stopped_by_done,
        "session_state": session_state,
    }
