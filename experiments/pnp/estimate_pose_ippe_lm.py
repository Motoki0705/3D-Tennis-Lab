#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path
import math

import cv2 as cv
import numpy as np
import yaml
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from types import SimpleNamespace


# -------------------------------
# I/O utilities
# -------------------------------
def load_camera_yaml(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Accepts YAML with either:
      - K as a 3x3 or fx, fy, cx, cy
      - dist as list (length 0,4,5,8,12,14 supported by OpenCV)
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "K" not in data:
        raise ValueError("camera yaml must contain 'K'.")

    Kv = data["K"]
    if isinstance(Kv, list):
        K = np.asarray(Kv, dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError("K provided as list must be 3x3")
    elif isinstance(Kv, dict):
        fx = float(Kv["fx"])
        fy = float(Kv["fy"])
        cx = float(Kv["cx"])
        cy = float(Kv["cy"])
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    else:
        raise ValueError("Unsupported K format in camera yaml.")

    dist = np.zeros((0, 1), dtype=np.float64)
    if "dist" in data:
        d = np.asarray(data["dist"], dtype=np.float64).reshape(-1, 1)
        # OpenCVの標準長のみ許可（0,4,5,8,12,14）
        if d.size not in (0, 4, 5, 8, 12, 14):
            raise ValueError(f"Unsupported distortion length: {d.size}")
        dist = d

    return K, dist


def load_court_spec(path: Path, force_z0: bool = False) -> dict[int, np.ndarray]:
    """
    Reads 'keypoints_3d_m' mapping "0".."14" -> [x,y,z].
    Returns {index: np.array([x,y,z], float32)}.
    """
    with open(path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    kp3d = spec.get("keypoints_3d_m", {})
    out = {}
    for k, v in kp3d.items():
        idx = int(k)
        xyz = np.asarray(v, dtype=np.float32)
        if xyz.shape != (3,):
            raise ValueError(f"Key {k} has invalid xyz shape: {xyz.shape}")
        if force_z0:
            xyz = xyz * np.array([1, 1, 0], dtype=np.float32)
        out[idx] = xyz
    if len(out) < 4:
        raise ValueError("Court spec must contain at least 4 keypoints.")
    return out


def parse_coco(coco_json: Path):
    """
    Returns:
        images_by_id: {image_id: {"file_name", "width", "height"}}
        ann_by_image: {image_id: annotation_dict}  # assumes single court per image
    """
    with open(coco_json, encoding="utf-8") as f:
        coco = json.load(f)

    images_by_id = {}
    for im in coco["images"]:
        images_by_id[im["id"]] = {
            "file_name": im["file_name"],
            "width": im["width"],
            "height": im["height"],
        }

    ann_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in ann_by_image:
            ann_by_image[img_id] = ann
        # 複数注釈があるならここでポリシー選択（例: 最大面積、特定category等）

    return images_by_id, ann_by_image


# -------------------------------
# Geometry & helpers
# -------------------------------
def extract_visible_correspondences(
    ann: dict,
    court3d: dict[int, np.ndarray],
    vmin: int = 2,
    kp_scale: float = 1.0,
) -> tuple[np.ndarray | None, np.ndarray | None, list[int]]:
    """
    From COCO keypoints [x,y,v ...], build objectPoints (Nx3) and imagePoints (Nx2)
    using only entries with visibility >= vmin.
    """
    kps = ann["keypoints"]
    if len(kps) % 3 != 0:
        raise ValueError("keypoints length is not a multiple of 3")
    n = len(kps) // 3
    obj_pts, img_pts, used_idx = [], [], []
    for i in range(n):
        x = kps[3 * i + 0]
        y = kps[3 * i + 1]
        v = kps[3 * i + 2]
        if v >= vmin and i in court3d:
            obj_pts.append(court3d[i])
            img_pts.append([x * kp_scale, y * kp_scale])
            used_idx.append(i)
    if not obj_pts:
        return None, None, []
    return (np.asarray(obj_pts, dtype=np.float32), np.asarray(img_pts, dtype=np.float32), used_idx)


def rodrigues_to_matrix(rvec):
    R, _ = cv.Rodrigues(rvec)
    return R


def world_from_camera(R_wc=None, t_wc=None, R_cw=None, t_cw=None):
    """
    Convert between world->camera (R_cw, t_cw) and camera->world (R_wc, t_wc).
    OpenCV returns world->camera: X_c = R_cw X_w + t_cw.
    """
    if R_cw is not None and t_cw is not None:
        R_wc = R_cw.T
        t_wc = -R_wc @ t_cw
        return R_wc, t_wc
    elif R_wc is not None and t_wc is not None:
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc
        return R_cw, t_cw
    else:
        raise ValueError("Provide either (R_cw, t_cw) or (R_wc, t_wc).")


def quat_xyzw_from_R(R: np.ndarray) -> tuple[float, float, float, float]:
    """
    Converts rotation matrix to quaternion (x,y,z,w).
    """
    # robust conversion
    K = (
        np.array([
            [R[0, 0] - R[1, 1] - R[2, 2], R[1, 0] + R[0, 1], R[2, 0] + R[0, 2], R[1, 2] - R[2, 1]],
            [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], R[2, 1] + R[1, 2], R[2, 0] - R[0, 2]],
            [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], R[0, 1] - R[1, 0]],
            [R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0], R[0, 0] + R[1, 1] + R[2, 2]],
        ])
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    q = eigvecs[:, np.argmax(eigvals)]
    x, y, z, w = q[0], q[1], q[2], q[3]
    # normalize
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n > 0:
        x, y, z, w = x / n, y / n, z / n, w / n
    return float(x), float(y), float(z), float(w)


def ypr_from_R(R: np.ndarray) -> tuple[float, float, float]:
    """
    yaw(Z), pitch(Y), roll(X) in degrees, from camera->world R_wc
    """
    # ZYX
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        pitch = math.degrees(math.atan2(-R[2, 0], sy))
        roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    else:
        yaw = math.degrees(math.atan2(-R[0, 1], R[1, 1]))
        pitch = math.degrees(math.atan2(-R[2, 0], sy))
        roll = 0.0
    return yaw, pitch, roll


def compute_reprojection_errors(obj_pts, img_pts, K, dist, rvec, tvec):
    proj, _ = cv.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    residuals = np.linalg.norm(proj - img_pts, axis=1)
    return residuals, proj


# -------------------------------
# Robust refinement (IRLS-like)
# -------------------------------
def refine_with_irls(obj_pts, img_pts, K, dist, rvec, tvec, iters: int = 3, tau_px: float = 3.0, min_pts: int = 4):
    """
    Iteratively reweighted (via shrinking threshold). Uses hard mask update (Huber近似の簡略化)。
    """
    mask = np.ones(len(obj_pts), dtype=bool)
    cur_tau = float(tau_px)
    for _ in range(max(1, iters)):
        if mask.sum() < min_pts:
            break
        rvec, tvec = cv.solvePnPRefineLM(obj_pts[mask], img_pts[mask], K, dist, rvec, tvec)
        residuals, _ = compute_reprojection_errors(obj_pts, img_pts, K, dist, rvec, tvec)
        mask = residuals < cur_tau
        cur_tau = max(1.0, 0.75 * cur_tau)  # 徐々に厳しく
    return rvec, tvec, mask


# -------------------------------
# Candidate selection & fallback
# -------------------------------
def choose_best_ippe_candidate_hard(
    obj_pts,
    img_pts,
    K,
    dist,
    rvecs,
    tvecs,
    reproj_errs=None,
    min_height_m=0.3,
    cheirality_frac=0.6,
    enforce_downward=True,
):
    n = len(rvecs)
    if n <= 1:
        return 0

    def score_tuple(rvec, tvec):
        proj, _ = cv.projectPoints(obj_pts, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)
        res = np.linalg.norm(proj - img_pts, axis=1)
        mean_e = float(res.mean())
        med_e = float(np.median(res))

        R_cw, _ = cv.Rodrigues(rvec)
        R_wc = R_cw.T
        t_wc = -R_wc @ tvec.reshape(3, 1)

        # cheirality
        Xc = (R_cw @ obj_pts.T + tvec.reshape(3, 1)).T
        cheir = (Xc[:, 2] > 0).mean()

        ok = True
        if float(t_wc[2, 0]) < min_height_m:
            ok = False
        if cheir < cheirality_frac:
            ok = False
        if enforce_downward and (R_wc[2, 2] >= 0.0):
            ok = False

        return ok, mean_e, med_e

    cand = []
    for i in range(n):
        ok, mean_e, med_e = score_tuple(rvecs[i], tvecs[i])
        # 優先: ok, 次: median誤差
        cand.append((i, ok, med_e, mean_e))

    valids = [c for c in cand if c[1]]
    if valids:
        return min(valids, key=lambda x: (x[2], x[3]))[0]
    else:
        return min(cand, key=lambda x: (x[2], x[3]))[0]


def fallback_h_decompose(obj_pts, img_pts, K, dist=None, min_height_m=0.3, cheirality_frac=0.6, enforce_downward=True):
    """
    Planar fallback using homography decomposition.
    obj_pts: Nx3 (Z≃0前提)
    """
    obj2d = obj_pts[:, :2].astype(np.float64)
    img2d = img_pts.astype(np.float64)

    # ホモグラフィ推定（RANSAC）
    H, _ = cv.findHomography(obj2d, img2d, method=cv.RANSAC, ransacReprojThreshold=3.0)
    if H is None:
        return None, None

    # 分解
    sols = cv.decomposeHomographyMat(H, K)
    if sols is None:
        return None, None
    nsol = sols[0]
    best = None
    for i in range(nsol):
        R_cw = sols[1][i]
        t_cw = sols[2][i]
        # 深度尺度不定 → tのスケールは後段で暗に吸収される
        rvec, _ = cv.Rodrigues(R_cw)
        # スコアリング
        _ = choose_best_ippe_candidate_hard(
            obj_pts,
            img_pts,
            K,
            dist,
            [rvec],
            [t_cw.reshape(3, 1)],
            reproj_errs=None,
            min_height_m=min_height_m,
            cheirality_frac=cheirality_frac,
            enforce_downward=enforce_downward,
        )
        # score関数を再利用できないため、ここで誤差計算
        proj, _ = cv.projectPoints(obj_pts, rvec, t_cw, K, dist)
        err = float(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1).median())
        if best is None or err < best[0]:
            best = (err, rvec, t_cw.reshape(3, 1))

    if best is None:
        return None, None
    return best[1], best[2]


# -------------------------------
# Intrinsics auto-tuning
# -------------------------------
def _grid_values(center, span, n):
    return np.linspace(center - span, center + span, n)


def auto_tune_intrinsics_per_image(
    obj_pts,
    img_pts,
    K0,
    dist,
    dc: float = 40.0,
    nc: int = 7,
    f_scale: float = 0.2,
    nf: int = 7,
    reuse_LM: bool = True,
):
    """
    Coarse grid search for (fx=fy), cx, cy with optional LM reuse。
    Returns (err, fx, cx, cy, rvec, tvec)
    """
    cx0, cy0 = float(K0[0, 2]), float(K0[1, 2])
    fx0 = float(K0[0, 0])

    cxs = _grid_values(cx0, dc, nc)
    cys = _grid_values(cy0, dc, nc)
    fxs = np.linspace(fx0 * (1 - f_scale), fx0 * (1 + f_scale), nf)

    best = None
    cached_pose = None  # reuse
    for fx in fxs:
        for cx in cxs:
            for cy in cys:
                Kt = K0.copy()
                Kt[0, 0] = fx
                Kt[1, 1] = fx
                Kt[0, 2] = cx
                Kt[1, 2] = cy

                if cached_pose is not None and reuse_LM:
                    r0, t0 = cached_pose
                    rvec, tvec = cv.solvePnPRefineLM(obj_pts, img_pts, Kt, dist, r0, t0)
                else:
                    ok, rvecs, tvecs, err = cv.solvePnPGeneric(obj_pts, img_pts, Kt, dist, flags=cv.SOLVEPNP_IPPE)
                    if not ok or len(rvecs) == 0:
                        # fallback
                        rvec, tvec = fallback_h_decompose(obj_pts, img_pts, Kt, dist)
                        if rvec is None:
                            continue
                    else:
                        idx = choose_best_ippe_candidate_hard(obj_pts, img_pts, Kt, dist, rvecs, tvecs, err)
                        rvec, tvec = rvecs[idx], tvecs[idx]

                    rvec, tvec = cv.solvePnPRefineLM(obj_pts, img_pts, Kt, dist, rvec, tvec)
                    cached_pose = (rvec, tvec)

                proj, _ = cv.projectPoints(obj_pts, rvec, tvec, Kt, dist)
                mean_e = float(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1).mean())
                rec = (mean_e, fx, cx, cy, rvec, tvec)
                if (best is None) or (mean_e < best[0]):
                    best = rec
    return best


def collect_samples_for_global(images_by_id, ann_by_image, court3d, sample_n: int, vmin: int, kp_scale: float):
    samples = []
    for img_id, iminfo in images_by_id.items():
        if img_id not in ann_by_image:
            continue
        obj_pts, img_pts, _ = extract_visible_correspondences(
            ann_by_image[img_id], court3d, vmin=vmin, kp_scale=kp_scale
        )
        if obj_pts is None or len(obj_pts) < 4:
            continue
        samples.append((img_id, iminfo, obj_pts, img_pts))
        if len(samples) >= sample_n:
            break
    return samples


def global_autotune_intrinsics(samples, K0, dist, dc=60.0, nc=7, f_scale=0.25, nf=7):
    """
    Choose K (fx=cx=cy pattern) minimizing mean error across samples.
    Returns (fx, cx, cy).
    """
    cx0, cy0 = float(K0[0, 2]), float(K0[1, 2])
    fx0 = float(K0[0, 0])

    cxs = _grid_values(cx0, dc, nc)
    cys = _grid_values(cy0, dc, nc)
    fxs = np.linspace(fx0 * (1 - f_scale), fx0 * (1 + f_scale), nf)

    best = None
    for fx in fxs:
        for cx in cxs:
            for cy in cys:
                Kt = K0.copy()
                Kt[0, 0] = fx
                Kt[1, 1] = fx
                Kt[0, 2] = cx
                Kt[1, 2] = cy

                errs = []
                for _, _, obj_pts, img_pts in samples:
                    ok, rvecs, tvecs, err = cv.solvePnPGeneric(obj_pts, img_pts, Kt, dist, flags=cv.SOLVEPNP_IPPE)
                    if not ok or len(rvecs) == 0:
                        rvec, tvec = fallback_h_decompose(obj_pts, img_pts, Kt, dist)
                        if rvec is None:
                            continue
                    else:
                        idx = choose_best_ippe_candidate_hard(obj_pts, img_pts, Kt, dist, rvecs, tvecs, err)
                        rvec, tvec = rvecs[idx], tvecs[idx]
                    # 1回LM
                    rvec, tvec = cv.solvePnPRefineLM(obj_pts, img_pts, Kt, dist, rvec, tvec)
                    proj, _ = cv.projectPoints(obj_pts, rvec, tvec, Kt, dist)
                    errs.append(float(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1).mean()))
                if not errs:
                    continue
                mean_err = float(np.mean(errs))
                if best is None or mean_err < best[0]:
                    best = (mean_err, fx, cx, cy)

    if best is None:
        return float(K0[0, 0]), float(K0[0, 2]), float(K0[1, 2])
    return best[1], best[2], best[3]


# -------------------------------
# Visualization
# -------------------------------
def draw_overlay(image, img_pts, proj_pts, used_idx, out_path, inlier_mask=None):
    """
    Draw GT keypoints (green), projected points (magenta for inliers / red for outliers), and index labels.
    """
    vis = image.copy()
    if len(vis.shape) == 2 or vis.shape[2] == 1:
        vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)

    N = len(used_idx)
    for j in range(N):
        gt = tuple(map(int, np.round(img_pts[j])))
        pr = tuple(map(int, np.round(proj_pts[j])))
        is_in = True if (inlier_mask is None or inlier_mask[j]) else False
        cv.circle(vis, gt, 4, (0, 255, 0), -1)  # GT
        cv.circle(vis, pr, 4, (255, 0, 255) if is_in else (0, 0, 255), -1)  # projected
        cv.line(vis, gt, pr, (128, 128, 128), 1)
        cv.putText(vis, str(used_idx[j]), gt, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)

    cv.imwrite(str(out_path), vis)


# -------------------------------
# Processing one image
# -------------------------------
def process_one(img_id, iminfo, ann, court3d, base_K, base_dist, args, K_global=None):
    # 対応点
    obj_pts, img_pts, used_idx = extract_visible_correspondences(ann, court3d, vmin=args.vmin, kp_scale=args.kp_scale)
    if obj_pts is None or len(obj_pts) < 4:
        return None

    w, h = int(iminfo["width"]), int(iminfo["height"])

    # intrinsics
    if base_K is None:
        # naive guess
        f_guess = 1.2 * max(w, h)
        fx = args.fx if args.fx is not None else f_guess
        fy = args.fy if args.fy is not None else f_guess
        cx = args.cx if args.cx is not None else w * 0.5
        cy = args.cy if args.cy is not None else h * 0.5
        K_use = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist_use = (
            np.zeros((0, 1), dtype=np.float64)
            if args.dist is None
            else np.asarray(args.dist, dtype=np.float64).reshape(-1, 1)
        )
    else:
        K_use = base_K.copy()
        dist_use = base_dist.copy() if base_dist is not None else np.zeros((0, 1), dtype=np.float64)

    # Global autotune (固定)
    if args.autotune_k == "global" and K_global is not None:
        K_use[0, 0] = K_use[1, 1] = K_global[0]
        K_use[0, 2], K_use[1, 2] = K_global[1], K_global[2]

    # undistort前正規化
    if args.use_undistort and dist_use.size > 0:
        img_ud = cv.undistortPoints(np.expand_dims(img_pts, 1), K_use, dist_use, P=K_use).squeeze(1)
        img_for_pnp, dist_for_pnp = img_ud, None
    else:
        img_for_pnp, dist_for_pnp = img_pts, dist_use

    # per-image autotune
    rvec0, tvec0 = None, None
    if args.autotune_k == "per-image":
        best = auto_tune_intrinsics_per_image(
            obj_pts, img_for_pnp, K_use, dist_for_pnp, dc=args.dc, nc=args.nc, f_scale=args.f_scale, nf=args.nf
        )
        if best is not None:
            _, fx, cx, cy, rvec0, tvec0 = best
            K_use[0, 0] = K_use[1, 1] = fx
            K_use[0, 2], K_use[1, 2] = cx, cy

    # ダイナミック既定のtau（px）
    tau_px = args.inlier_px
    if tau_px is None or tau_px <= 0:
        tau_px = 0.005 * math.hypot(w, h)

    # IPPE初期化
    retval, rvecs, tvecs, reproj_errors = cv.solvePnPGeneric(
        obj_pts, img_for_pnp, K_use, dist_for_pnp, flags=cv.SOLVEPNP_IPPE
    )
    if retval and len(rvecs) > 0:
        idx = choose_best_ippe_candidate_hard(
            obj_pts,
            img_pts,
            K_use,
            dist_use,
            rvecs,
            tvecs,
            reproj_errors,
            min_height_m=args.min_height,
            cheirality_frac=args.cheirality_frac,
            enforce_downward=not args.allow_upward,
        )
        rvec, tvec = rvecs[idx], tvecs[idx]
    elif rvec0 is not None:
        rvec, tvec = rvec0, tvec0
    else:
        # 平面に合致するフォールバック
        rvec, tvec = fallback_h_decompose(
            obj_pts,
            img_pts,
            K_use,
            dist_use,
            min_height_m=args.min_height,
            cheirality_frac=args.cheirality_frac,
            enforce_downward=not args.allow_upward,
        )
        if rvec is None:
            return None

    # LM + IRLS
    rvec, tvec = cv.solvePnPRefineLM(obj_pts, img_pts, K_use, dist_use, rvec, tvec)
    rvec, tvec, inlier_mask = refine_with_irls(
        obj_pts, img_pts, K_use, dist_use, rvec, tvec, iters=args.irls_iters, tau_px=tau_px, min_pts=4
    )

    # 終値の誤差
    residuals, proj = compute_reprojection_errors(obj_pts, img_pts, K_use, dist_use, rvec, tvec)
    mean_err = float(residuals.mean())
    median_err = float(np.median(residuals))
    p90 = float(np.percentile(residuals, 90))
    p95 = float(np.percentile(residuals, 95))
    inlier_rate = float(np.mean(inlier_mask.astype(np.float32)))

    # 変換表現
    R_cw = rodrigues_to_matrix(rvec)
    t_cw = tvec.reshape(3, 1)
    R_wc, t_wc = world_from_camera(R_cw=R_cw, t_cw=t_cw)
    quat_xyzw = quat_xyzw_from_R(R_wc)
    yaw, pitch, roll = ypr_from_R(R_wc)

    # 可視化
    vis_path = None
    if args.visualize:
        # 画像が存在する場合のみ
        img_path = Path(args.images_dir) / iminfo["file_name"]
        if img_path.exists():
            image = cv.imread(str(img_path), cv.IMREAD_COLOR)
            if image is not None:
                vis_dir = Path(args.out_dir) / "vis"
                vis_dir.mkdir(parents=True, exist_ok=True)
                vis_path = vis_dir / iminfo["file_name"]
                draw_overlay(image, img_pts, proj, used_idx, vis_path, inlier_mask=inlier_mask)

    # 出力レコード
    rec = {
        "image_id": int(img_id),
        "file_name": iminfo["file_name"],
        "width": w,
        "height": h,
        "used_indices": [int(i) for i in used_idx],
        "inlier_mask": [bool(x) for x in inlier_mask.tolist()],
        "inlier_rate": inlier_rate,
        "mean_reproj_px": mean_err,
        "median_reproj_px": median_err,
        "p90_reproj_px": p90,
        "p95_reproj_px": p95,
        "rvec_cw": rvec.reshape(-1).tolist(),
        "tvec_cw": t_cw.reshape(-1).tolist(),
        "R_cw": R_cw.reshape(-1).tolist(),
        "R_wc": R_wc.reshape(-1).tolist(),
        "t_wc": t_wc.reshape(-1).tolist(),
        "quat_xyzw": list(map(float, quat_xyzw)),
        "ypr_deg": [yaw, pitch, roll],
        "K": K_use.reshape(-1).tolist(),
        "dist": dist_use.reshape(-1).tolist(),
        "visualization": str(vis_path) if vis_path is not None else None,
    }
    return rec


# -------------------------------
# Main
# -------------------------------
@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Hydraベースのエントリポイント。
    既存の補助関数（load_camera_yaml, load_court_spec, parse_coco,
    collect_samples_for_global, global_autotune_intrinsics, process_one）をそのまま利用します。
    """

    # --- パス解決（Hydraは作業ディレクトリをrunディレクトリに変更するため、入力系は絶対化） ---
    images_dir = Path(to_absolute_path(cfg.paths.images_dir))
    coco_json = Path(to_absolute_path(cfg.paths.coco_json))
    court_spec = Path(to_absolute_path(cfg.paths.court_spec))
    out_dir = Path(to_absolute_path(cfg.paths.out_dir))
    camera_yaml_path = None
    if cfg.paths.camera_yaml is not None and str(cfg.paths.camera_yaml) != "":
        camera_yaml_path = Path(to_absolute_path(cfg.paths.camera_yaml))

    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.exec.visualize:
        (out_dir / "vis").mkdir(parents=True, exist_ok=True)

    # --- カメラ内参の初期化 ---
    if cfg.intrinsics.use_camera_yaml and camera_yaml_path is not None:
        base_K, base_dist = load_camera_yaml(camera_yaml_path)
    else:
        base_K, base_dist = None, None  # process_one 側で画像サイズから推定/CLI相当の値を適用

    # --- コート仕様とCOCO ---
    court3d = load_court_spec(court_spec, force_z0=cfg.filter.force_z0)
    images_by_id, ann_by_image = parse_coco(coco_json)

    # --- Globalオートチューン（必要時） ---
    K_global = None
    if cfg.autotune.mode == "global":
        # 初期Kを準備（camera_yaml が無い場合は先頭画像サイズから推定）
        if base_K is None:
            any_im = next(iter(images_by_id.values()))
            w0, h0 = int(any_im["width"]), int(any_im["height"])
            f0 = 1.2 * max(w0, h0)
            init_K = np.array([[f0, 0, w0 * 0.5], [0, f0, h0 * 0.5], [0, 0, 1]], dtype=np.float64)
            init_dist = np.zeros((0, 1), dtype=np.float64)
        else:
            init_K = base_K.copy()
            init_dist = base_dist.copy() if base_dist is not None else np.zeros((0, 1), dtype=np.float64)

        samples = collect_samples_for_global(
            images_by_id,
            ann_by_image,
            court3d,
            sample_n=int(cfg.autotune.sample),
            vmin=int(cfg.filter.vmin),
            kp_scale=float(cfg.filter.kp_scale),
        )
        if samples:
            fx, cx, cy = global_autotune_intrinsics(
                samples,
                init_K,
                init_dist,
                dc=float(cfg.autotune.dc),
                nc=int(cfg.autotune.nc),
                f_scale=float(cfg.autotune.f_scale),
                nf=int(cfg.autotune.nf),
            )
            K_global = (fx, cx, cy)
            print(f"[global-autotune] fx=fy={fx:.3f}, cx={cx:.3f}, cy={cy:.3f}")
        else:
            print("[global-autotune] No valid samples; skipping.")

    # --- 既存process_oneが期待する名前で設定をまとめる（argparse代替のコンテナ） ---
    args = SimpleNamespace(
        # paths
        images_dir=str(images_dir),
        out_dir=str(out_dir),
        # visibility / plane / kp scale
        vmin=int(cfg.filter.vmin),
        force_z0=bool(cfg.filter.force_z0),
        kp_scale=float(cfg.filter.kp_scale),
        # robust
        inlier_px=(None if cfg.robust.inlier_px is None else float(cfg.robust.inlier_px)),
        irls_iters=int(cfg.robust.irls_iters),
        # physical constraints
        min_height=float(cfg.constraints.min_height),
        cheirality_frac=float(cfg.constraints.cheirality_frac),
        allow_upward=bool(cfg.constraints.allow_upward),
        # autotune
        autotune_k=str(cfg.autotune.mode),
        autotune_sample=int(cfg.autotune.sample),
        dc=float(cfg.autotune.dc),
        nc=int(cfg.autotune.nc),
        f_scale=float(cfg.autotune.f_scale),
        nf=int(cfg.autotune.nf),
        # undistort
        use_undistort=bool(cfg.undistort.use),
        # execution
        limit=(None if cfg.exec.limit is None else int(cfg.exec.limit)),
        visualize=bool(cfg.exec.visualize),
        jobs=int(cfg.exec.jobs),
        # intrinsics直指定（camera_yamlが無い場合に参照）
        fx=(None if cfg.intrinsics.fx is None else float(cfg.intrinsics.fx)),
        fy=(None if cfg.intrinsics.fy is None else float(cfg.intrinsics.fy)),
        cx=(None if cfg.intrinsics.cx is None else float(cfg.intrinsics.cx)),
        cy=(None if cfg.intrinsics.cy is None else float(cfg.intrinsics.cy)),
        dist=([] if cfg.intrinsics.dist is None else list(cfg.intrinsics.dist)),
    )

    # --- 対象画像の列挙 ---
    items = []
    for img_id, iminfo in images_by_id.items():
        if args.limit is not None and len(items) >= args.limit:
            break
        if img_id not in ann_by_image:
            continue
        items.append((img_id, iminfo, ann_by_image[img_id]))

    # --- 推定・保存 ---
    out_jsonl = open(Path(out_dir) / "poses.jsonl", "w", encoding="utf-8")
    processed = 0

    if args.jobs <= 1:
        for img_id, iminfo, ann in items:
            rec = process_one(img_id, iminfo, ann, court3d, base_K, base_dist, args, K_global=K_global)
            if rec is None:
                continue
            out_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
            processed += 1
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = [
                ex.submit(process_one, img_id, iminfo, ann, court3d, base_K, base_dist, args, K_global)
                for img_id, iminfo, ann in items
            ]
            for fu in as_completed(futs):
                rec = fu.result()
                if rec is None:
                    continue
                out_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                processed += 1

    out_jsonl.close()
    print(f"Processed images: {processed}")
    print(f"Results written to: {Path(out_dir) / 'poses.jsonl'}")
    if args.visualize:
        print(f"Overlays in: {Path(out_dir) / 'vis'}")


if __name__ == "__main__":
    main()
