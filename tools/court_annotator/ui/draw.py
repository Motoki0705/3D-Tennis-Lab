import cv2


def _put_text(
    bgr_frame,
    text,
    org,
    font_face=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.6,
    color=(255, 255, 255),
    thickness=1,
    shadow_color=(0, 0, 0),
):
    """Draws text with a thicker outline for better visibility."""
    (w, _), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
    # Strong multi-directional shadow/outline
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (1, -1), (-1, 1)]:
        cv2.putText(
            bgr_frame, text, (org[0] + dx, org[1] + dy), font_face, font_scale, shadow_color, thickness + 2, cv2.LINE_AA
        )
    cv2.putText(bgr_frame, text, org, font_face, font_scale, color, thickness + 1, cv2.LINE_AA)
    return org[0] + w, org[1]


def draw_skeleton(bgr_frame, kps, skeleton, colors):
    """Draws the skeleton lines on the frame, color-coded by source."""
    for i, j in skeleton:
        pt1 = kps[i]
        pt2 = kps[j]
        if pt1.v > 0 and pt2.v > 0 and pt1.x is not None and pt2.x is not None:
            is_auto = pt1.source == "auto" or pt2.source == "auto"
            color = colors.get("line_auto") if is_auto else colors.get("line_user")
            thickness = 1 if is_auto else 2
            cv2.line(bgr_frame, (int(pt1.x), int(pt1.y)), (int(pt2.x), int(pt2.y)), color, thickness, cv2.LINE_AA)


def draw_points(bgr_frame, kps, colors, focus_idx=None):
    """Draws the keypoints on the frame, color-coded by source and visibility."""
    for i, kp in enumerate(kps):
        if kp.v > 0 and kp.x is not None:
            if kp.source == "user":
                color = colors.get("point_user_visible") if kp.v == 2 else colors.get("point_user_occluded")
            else:  # auto
                color = colors.get("point_auto")

            cv2.circle(bgr_frame, (int(kp.x), int(kp.y)), 5, color, -1)
            cv2.circle(bgr_frame, (int(kp.x), int(kp.y)), 6, (30, 30, 30), 1, cv2.LINE_AA)

    # Highlight the focused point with a larger circle if it's not placed yet
    if focus_idx is not None and kps[focus_idx].v == 0:
        # This is for the cursor-following focus, which is not implemented yet.
        # We will show focus in the HUD instead.
        pass


def draw_hud(bgr_frame, bundle, colors):
    """Draws the Heads-Up Display."""
    frame_idx = bundle.get("frame_idx", 0)
    nframes = bundle.get("nframes", 0)
    is_ready = bundle.get("is_ready", False)
    fit_result = bundle.get("fit_result")
    frame_state = bundle.get("frame_state")
    focus_idx = bundle.get("focus_idx", 0)
    court_spec = bundle.get("court_spec")

    # --- Top Left Info ---
    y_pos = 30
    _put_text(bgr_frame, f"Frame: {frame_idx:06d} / {nframes}", (10, y_pos), color=colors.get("hud_text"))
    y_pos += 25

    if fit_result and fit_result.H is not None:
        rmse_str = f"{fit_result.rmse:.2f}" if fit_result.rmse is not None else "n/a"
        _put_text(
            bgr_frame, f"H | Used: {fit_result.used} | RMSE: {rmse_str} px", (10, y_pos), color=colors.get("hud_text")
        )
    else:
        _put_text(bgr_frame, "H: n/a", (10, y_pos), color=colors.get("hud_text"))
    y_pos += 25

    if is_ready:
        _put_text(bgr_frame, "READY", (10, y_pos), color=colors.get("ready_text"), font_scale=0.7, thickness=2)
    else:
        _put_text(bgr_frame, "NOT READY", (10, y_pos), color=(50, 50, 220), font_scale=0.7, thickness=2)

    # --- Top Right Info ---
    if court_spec:
        focus_name = court_spec.names[focus_idx]
        _put_text(
            bgr_frame,
            f"Focus [{focus_idx}]: {focus_name}",
            (bgr_frame.shape[1] - 500, 30),
            color=colors.get("hud_text"),
        )

    if frame_state:
        _put_text(
            bgr_frame,
            f"Locked: {sorted(list(frame_state.locked))}",
            (bgr_frame.shape[1] - 500, 55),
            color=colors.get("hud_text"),
        )
        _put_text(
            bgr_frame,
            f"Skipped: {sorted(list(frame_state.skipped))}",
            (bgr_frame.shape[1] - 500, 80),
            color=colors.get("hud_text"),
        )

    # --- Bottom Center Message ---
    hud_message = bundle.get("hud_message")
    if hud_message:
        _put_text(
            bgr_frame,
            hud_message,
            (bgr_frame.shape[1] // 2 - 100, bgr_frame.shape[0] - 20),
            color=(200, 255, 200),
            font_scale=0.7,
            thickness=2,
        )


def render(bundle):
    """Main rendering function."""
    frame = bundle["frame"].copy()
    cfg = bundle.get("config", {})
    ui_cfg = cfg.get("ui", {})
    colors = ui_cfg.get("colors", {})

    frame_state = bundle.get("frame_state")
    court_spec = bundle.get("court_spec")
    focus_idx = bundle.get("focus_idx")

    if frame_state and court_spec:
        draw_points(frame, frame_state.kps, colors, focus_idx)
        draw_skeleton(frame, frame_state.kps, court_spec.skeleton, colors)

    draw_hud(frame, bundle, colors)

    bundle["rendered_frame"] = frame
    return bundle
