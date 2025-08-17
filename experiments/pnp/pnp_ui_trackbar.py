#!/usr/bin/env python3
"""
Interactive UI (OpenCV Trackbars) to tweak camera/court R,t and project tennis-court keypoints.
- Uses OpenCV windows & trackbars (no ipywidgets required)
- Draws 2D projected points with numeric IDs
- Supports background image or blank canvas
- Intrinsics: fx, fy, cx, cy + optional distortion
- Extrinsics: camera yaw/pitch/roll (deg), tx,ty,tz (m)
- Court extra transform: yaw/pitch/roll, tx,ty,tz
- Mirror Y toggle to quickly check left/right mapping
- Save overlay: press 's'  |  Quit: 'q'
"""

import argparse
import os
import textwrap
from dataclasses import dataclass

import cv2
import numpy as np
import yaml

# ------------------- Core helpers -------------------


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def K(self) -> np.ndarray:
        K = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        return K


def parse_dist_coeffs(s: str | None) -> np.ndarray | None:
    if s is None:
        return None
    s = s.strip()
    if s.lower() in ["", "none", "null"]:
        return None
    vals = [float(x) for x in s.split(",")]
    if len(vals) not in (4, 5, 8):
        errer = "dist-coeffs must be 4, 5, or 8 floats"
        raise ValueError(errer)
    return np.array(vals, dtype=np.float64).reshape(-1, 1)


def project_points(pts3d: np.ndarray, K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, dist=None):
    pts2d_proj, _ = cv2.projectPoints(
        objectPoints=pts3d.astype(np.float64),
        rvec=rvec.astype(np.float64),
        tvec=tvec.astype(np.float64),
        cameraMatrix=K.astype(np.float64),
        distCoeffs=dist,
    )
    return pts2d_proj.reshape(-1, 2)


def rpy_deg_to_rvec(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    z = np.deg2rad(yaw_deg)
    y = np.deg2rad(pitch_deg)
    x = np.deg2rad(roll_deg)
    Rz, _ = cv2.Rodrigues(np.array([[0.0], [0.0], [z]], dtype=np.float64))
    Ry, _ = cv2.Rodrigues(np.array([[0.0], [y], [0.0]], dtype=np.float64))
    Rx, _ = cv2.Rodrigues(np.array([[x], [0.0], [0.0]], dtype=np.float64))
    R = Rz @ Ry @ Rx
    rvec, _ = cv2.Rodrigues(R)
    return rvec


def apply_court_transform(pts3d: np.ndarray, court_rvec: np.ndarray, court_tvec: np.ndarray) -> np.ndarray:
    Rc, _ = cv2.Rodrigues(court_rvec.astype(np.float64).reshape(3, 1))
    tc = court_tvec.astype(np.float64).reshape(3, 1)
    X = pts3d.astype(np.float64).T  # (3,N)
    Xp = Rc @ X + tc  # (3,N)
    return Xp.T  # (N,3)


def compose_extrinsics(cam_rvec, cam_tvec, court_rvec, court_tvec):
    Rc, _ = cv2.Rodrigues(court_rvec.astype(np.float64).reshape(3, 1))
    Rcam, _ = cv2.Rodrigues(cam_rvec.astype(np.float64).reshape(3, 1))
    tcam = cam_tvec.astype(np.float64).reshape(3, 1)
    tc = court_tvec.astype(np.float64).reshape(3, 1)
    R_new = Rcam @ Rc
    t_new = Rcam @ tc + tcam
    rvec_new, _ = cv2.Rodrigues(R_new)
    return rvec_new, t_new


DEFAULT_COURT_SPEC = textwrap.dedent("""
units: meters
axes:
  x: "length (+X toward far baseline)"
  y: "width  (+Y to the right when facing +X)"
  z: "up"
dimensions:
  half_length: 11.885
  half_singles: 4.115
  half_doubles: 5.4864
  service_from_net: 6.40
keypoints_3d_m:
  "0": [ 11.885,  5.4864, 0.0 ]
  "1": [ 11.885, -5.4864, 0.0 ]
  "2": [-11.885,  5.4864, 0.0 ]
  "3": [-11.885, -5.4864, 0.0 ]
  "4": [ 11.885,  4.115,  0.0 ]
  "6": [ 11.885, -4.115,  0.0 ]
  "5": [-11.885,  4.115,  0.0 ]
  "7": [-11.885, -4.115,  0.0 ]
  "8": [  6.40,   4.115,  0.0 ]
  "9": [  6.40,  -4.115,  0.0 ]
  "10": [-6.40,   4.115,  0.0 ]
  "11": [-6.40,  -4.115,  0.0 ]
  "12": [  6.40,   0.0,    0.0 ]
  "13": [ -6.40,   0.0,    0.0 ]
  "14": [  0.0,    0.0,    0.0 ]
""")


def load_court_spec(path_or_text: str | None) -> dict[str, tuple[float, float, float]]:
    if path_or_text is None:
        spec = yaml.safe_load(DEFAULT_COURT_SPEC)
    else:
        if os.path.isfile(path_or_text):
            with open(path_or_text, encoding="utf-8") as f:
                spec = yaml.safe_load(f)
        else:
            # assume raw YAML text
            spec = yaml.safe_load(path_or_text)
    name2xyz = spec.get("keypoints_3d_m", {})
    parsed = {}
    for k, v in name2xyz.items():
        if v is None:
            continue
        x, y, z = v
        parsed[str(k)] = (float(x), float(y), float(z))
    return parsed


# ------------------- UI helpers -------------------


def add_trackbar(win, name, minv, maxv, init, factor=1.0):
    cv2.createTrackbar(name, win, 0, int((maxv - minv) / factor), lambda v: None)
    cv2.setTrackbarPos(name, win, int((init - minv) / factor))
    return (minv, factor)  # store min and factor to recover value


def get_trackbar_value(win, name, min_factor_tuple):
    minv, factor = min_factor_tuple
    pos = cv2.getTrackbarPos(name, win)
    return minv + pos * factor


def main():  # noqa: C901
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--court-spec",
        type=str,
        default=None,
        help="Path to YAML (or YAML text). If omitted, uses default numeric spec.",
    )
    ap.add_argument("--image", type=str, default=None, help="Background image (optional)")
    ap.add_argument("--size", type=str, default="1920,1080", help="Canvas W,H when no image")
    ap.add_argument("--dist", type=str, default="", help='Distortion "k1,k2,p1,p2[,k3[,k4,k5,k6]]"')
    ap.add_argument(
        "--compose",
        type=str,
        default="points",
        choices=["points", "extrinsics"],
        help="Apply court transform to points or compose into extrinsics",
    )
    args = ap.parse_args()

    # Background
    if args.image and os.path.isfile(args.image):
        img0 = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if img0 is None:
            raise FileNotFoundError(args.image)
        H, W = img0.shape[:2]
    else:
        W, H = (int(s) for s in args.size.split(","))
        img0 = np.ones((H, W, 3), dtype=np.uint8) * 255

    # Spec
    name2xyz = load_court_spec(args.court_spec)
    ids = [str(i) for i in range(15) if str(i) in name2xyz]
    pts3d_base = np.array([name2xyz[i] for i in ids], dtype=np.float64)

    # Window
    win = "PnP Court UI (s: save, m: mirror Y, c: compose toggle, q: quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    # Trackbars
    # Intrinsics
    fx_tf = add_trackbar(win, "fx", 50.0, 6000.0, 1600.0, factor=10.0)
    fy_tf = add_trackbar(win, "fy", 50.0, 6000.0, 1600.0, factor=10.0)
    cx_tf = add_trackbar(win, "cx", 0.0, float(W), float(W) / 2, factor=1.0)
    cy_tf = add_trackbar(win, "cy", 0.0, float(H), float(H) / 2, factor=1.0)

    # Camera extrinsics
    yaw_tf = add_trackbar(win, "cam_yaw_deg", -180.0, 180.0, 0.0, factor=0.5)
    pitch_tf = add_trackbar(win, "cam_pitch_deg", -180.0, 180.0, 0.0, factor=0.5)
    roll_tf = add_trackbar(win, "cam_roll_deg", -180.0, 180.0, 0.0, factor=0.5)
    tx_tf = add_trackbar(win, "cam_tx(m)", -30.0, 30.0, 0.0, factor=0.05)
    ty_tf = add_trackbar(win, "cam_ty(m)", -30.0, 30.0, 0.0, factor=0.05)
    tz_tf = add_trackbar(win, "cam_tz(m)", -30.0, 60.0, 20.0, factor=0.05)

    # Court transform
    cyaw_tf = add_trackbar(win, "court_yaw_deg", -180.0, 180.0, 0.0, factor=0.5)
    cpitch_tf = add_trackbar(win, "court_pitch_deg", -180.0, 180.0, 0.0, factor=0.5)
    croll_tf = add_trackbar(win, "court_roll_deg", -180.0, 180.0, 0.0, factor=0.5)
    ctx_tf = add_trackbar(win, "court_tx(m)", -10.0, 10.0, 0.0, factor=0.01)
    cty_tf = add_trackbar(win, "court_ty(m)", -10.0, 10.0, 0.0, factor=0.01)
    ctz_tf = add_trackbar(win, "court_tz(m)", -10.0, 10.0, 0.0, factor=0.01)

    mirror_y = False
    compose_mode = args.compose  # 'points' or 'extrinsics'
    dist = parse_dist_coeffs(args.dist)

    save_count = 0

    while True:
        img = img0.copy()

        # Read trackbars
        fx = get_trackbar_value(win, "fx", fx_tf)
        fy = get_trackbar_value(win, "fy", fy_tf)
        cx = get_trackbar_value(win, "cx", cx_tf)
        cy = get_trackbar_value(win, "cy", cy_tf)

        yaw = get_trackbar_value(win, "cam_yaw_deg", yaw_tf)
        pitch = get_trackbar_value(win, "cam_pitch_deg", pitch_tf)
        roll = get_trackbar_value(win, "cam_roll_deg", roll_tf)
        tx = get_trackbar_value(win, "cam_tx(m)", tx_tf)
        ty = get_trackbar_value(win, "cam_ty(m)", ty_tf)
        tz = get_trackbar_value(win, "cam_tz(m)", tz_tf)

        cyaw = get_trackbar_value(win, "court_yaw_deg", cyaw_tf)
        cpitch = get_trackbar_value(win, "court_pitch_deg", cpitch_tf)
        croll = get_trackbar_value(win, "court_roll_deg", croll_tf)
        ctx = get_trackbar_value(win, "court_tx(m)", ctx_tf)
        cty = get_trackbar_value(win, "court_ty(m)", cty_tf)
        ctz = get_trackbar_value(win, "court_tz(m)", ctz_tf)

        K = Intrinsics(fx, fy, cx, cy).K()

        cam_rvec = rpy_deg_to_rvec(yaw, pitch, roll)
        cam_tvec = np.array([[tx], [ty], [tz]], dtype=np.float64)

        court_rvec = rpy_deg_to_rvec(cyaw, cpitch, croll)
        court_tvec = np.array([[ctx], [cty], [ctz]], dtype=np.float64)

        pts3d = pts3d_base.copy()
        if mirror_y:
            pts3d[:, 1] *= -1.0

        if compose_mode == "points":
            pts3d_use = apply_court_transform(pts3d, court_rvec, court_tvec)
            rvec_use, tvec_use = cam_rvec, cam_tvec
        else:
            pts3d_use = pts3d
            rvec_use, tvec_use = compose_extrinsics(cam_rvec, cam_tvec, court_rvec, court_tvec)

        pts2d = project_points(pts3d_use, K, rvec_use, tvec_use, dist)

        # Draw with IDs
        for pid, (u, v) in zip(ids, pts2d, strict=False):
            u_i, v_i = (round(u)), (round(v))
            cv2.circle(img, (u_i, v_i), 5, (0, 200, 0), -1)
            cv2.putText(img, str(pid), (u_i + 6, v_i - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(
                img, str(pid), (u_i + 6, v_i - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
            )

        # HUD text
        cv2.putText(
            img, str(pid), (u_i + 6, v_i - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA
        )  # 黒縁(太め)
        cv2.putText(
            img, str(pid), (u_i + 6, v_i - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA
        )  # 白文字(太さ2)

        cv2.imshow(win, img)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            os.makedirs("overlays", exist_ok=True)
            out_path = os.path.join("overlays", f"projection_{save_count:03d}.png")
            cv2.imwrite(out_path, img)
            print(f"[saved] {out_path}")
            save_count += 1
        elif key == ord("m"):
            mirror_y = not mirror_y
            print(f"[mirror Y] -> {mirror_y}")
        elif key == ord("c"):
            compose_mode = "extrinsics" if compose_mode == "points" else "points"
            print(f"[compose mode] -> {compose_mode}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
