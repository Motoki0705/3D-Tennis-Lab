#!/usr/bin/env python3

"""
PnP demo for Tennis Court keypoints (COCO-format)

Key features:
- Robust OpenCV PnP with RANSAC and optional refinement
- IPPE (planar) branch using solvePnPGeneric
- Distortion coefficients support
- Auto index→name mapping: aligns to court_spec keys if COCO names are numeric/empty/mismatched
- Optional left/right mirror retry and selection by reprojection error

Usage:
    python pnp_demo.py \
        --ann data/processed/court/annotation.json \
        --img-root data/processed/court/images \
        --court-spec court_spec.yaml \
        --out out_pnp_demo \
        --name-source auto \
        --max-images 50 \
        --fx 1600 --fy 1600 --cx auto --cy auto \
        --ransac-reproj-thresh 5.0 \
        --method EPNP \
        --refine \
        --try-mirror  # 既定True。無効にするには --no-try-mirror

Dependencies:
    pip install opencv-python-headless numpy pyyaml pandas
"""

import argparse
import ast
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import yaml

# ---- Canonical order: 数値IDで固定(specのキー "0".."14" と一致) ----
CANONICAL_ORDER = [str(i) for i in range(15)]


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
    if s is None or s.strip().lower() in ["", "none", "null"]:
        return None
    vals = [float(x) for x in s.split(",")]
    if len(vals) not in (4, 5, 8):
        errer = "dist-coeffs must be 4, 5, or 8 floats"
        raise ValueError(errer)
    return np.array(vals, dtype=np.float64).reshape(-1, 1)


def parse_args():
    ap = argparse.ArgumentParser(description="PnP demo for Tennis Court keypoints (COCO-format)")
    ap.add_argument("--ann", type=str, required=True, help="Path to COCO annotation.json")
    ap.add_argument("--img-root", type=str, required=True, help="Path to image root directory")
    ap.add_argument("--court-spec", type=str, default="court_spec.yaml", help="3D keypoint spec YAML")
    ap.add_argument("--out", type=str, default="out_pnp_demo", help="Output directory")
    ap.add_argument("--max-images", type=int, default=50, help="Process at most N images (0=all)")

    # intrinsics
    ap.add_argument("--fx", type=float, default=1600.0)
    ap.add_argument("--fy", type=float, default=1600.0)
    ap.add_argument("--cx", type=str, default="auto")
    ap.add_argument("--cy", type=str, default="auto")
    ap.add_argument(
        "--dist-coeffs",
        type=str,
        default=None,
        help='Comma-separated distortion coeffs "k1,k2,p1,p2[,k3[,k4,k5,k6]]"; omit or "none" for no distortion.',
    )

    ap.add_argument("--method", type=str, default="EPNP", choices=["EPNP", "ITERATIVE", "P3P", "AP3P", "IPPE"])
    ap.add_argument("--ransac-reproj-thresh", type=float, default=5.0, help="RANSAC reprojection threshold (px)")
    ap.add_argument(
        "--refine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refine result with ITERATIVE on inliers (default: True)",
    )

    # name resolution
    ap.add_argument(
        "--name-source",
        type=str,
        default="auto",
        choices=["auto", "coco", "spec"],
        help="How to resolve 2D index -> keypoint name mapping: "
        "'coco' uses categories[0].keypoints, "
        "'spec' aligns to court_spec keys, "
        "'auto' picks based on data (default).",
    )

    # mirror retry
    ap.add_argument(
        "--try-mirror",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Try y->-y mirrored 3D as a second hypothesis and pick lower reprojection error (default: True)",
    )

    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def load_coco(ann_path: str):
    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)
    images = {im["id"]: im for im in coco["images"]}
    anns_per_img = {}
    for ann in coco["annotations"]:
        if ann.get("category_id", 1) != 1:
            continue
        anns_per_img.setdefault(ann["image_id"], []).append(ann)
    if "categories" not in coco or len(coco["categories"]) == 0:
        kp_names = []
    else:
        kp_names = coco["categories"][0].get("keypoints", [])
    return images, anns_per_img, kp_names


def load_court_spec(path: str) -> dict[str, tuple[float, float, float]]:
    with open(path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    name2xyz = spec.get("keypoints_3d_m", {})

    # Allow simple arithmetic strings inside coordinate entries
    def f(e):
        if isinstance(e, int | float):
            return float(e)
        if isinstance(e, str):
            if not all(c in "0123456789.+-() " for c in e):
                errer = f"Unsafe expression in court_spec: {e}"
                raise ValueError(errer)
            return float(ast.literal_eval(e))
        return float(e)

    parsed = {}
    for k, v in name2xyz.items():
        if v is None:
            continue
        x, y, z = v
        parsed[str(k)] = (f(x), f(y), f(z))
    return parsed


def resolve_names(kp_names_from_coco: list[str], name_source: str, spec_keys: list[str]) -> list[str]:
    """
    返り値は「2Dインデックスi→この名前を3Dキーに使う」という配列。
    - 'spec'     : spec側のキーに整列(数値なら 0..14 の昇順など)
    - 'coco'     : COCOの names をそのまま使用
    - 'auto'     : COCO名が空/数値的/不一致なら specキーへフォールバック
    """

    def sort_like_spec(keys: list[str]) -> list[str]:
        # すべて整数変換できるなら数値昇順、できないならそのまま
        try:
            return sorted(keys, key=lambda s: int(s))
        except Exception:
            return list(keys)

    spec_keys_sorted = sort_like_spec(spec_keys)

    if name_source == "spec":
        return spec_keys_sorted

    if name_source == "coco":
        return [str(x) for x in kp_names_from_coco]

    # auto
    names = [str(x) for x in kp_names_from_coco] if kp_names_from_coco else []
    if not names:
        print("[INFO] No keypoint names in COCO; using spec keys.")
        return spec_keys_sorted

    def is_numeric_label(s: str) -> bool:
        s = s.strip()
        return s.isdigit() or (s.startswith("kp_") and s[3:].isdigit())

    if all(is_numeric_label(n) for n in names):
        print("[INFO] COCO keypoint names look numeric; aligning to spec keys.")
        return spec_keys_sorted

    # COCO名があるが、specキーとほぼ合っていない場合は spec にフォールバック
    if not any(n in spec_keys for n in names):
        print("[WARN] COCO names do not match spec keys; falling back to spec keys.")
        return spec_keys_sorted

    return names


def build_correspondences(ann: dict, resolved_names: list[str], name2xyz: dict[str, tuple[float, float, float]]):
    kps = ann["keypoints"]
    if len(kps) % 3 != 0:
        errer = "Keypoints length must be a multiple of 3"
        raise ValueError(errer)
    n = len(kps) // 3
    pts2d = []
    pts3d = []
    used_names = []
    missing = []
    for i in range(n):
        x = kps[3 * i + 0]
        y = kps[3 * i + 1]
        v = kps[3 * i + 2]
        if v <= 0:
            continue
        name = resolved_names[i] if i < len(resolved_names) else f"kp_{i}"
        if name not in name2xyz:
            missing.append(name)
            continue
        X = name2xyz[name]
        pts2d.append([x, y])
        pts3d.append([X[0], X[1], X[2]])
        used_names.append(name)
    if missing and len(used_names) == 0:
        print(f"[WARN] All visible keypoints missing in court_spec: {set(missing)}")
    if len(pts2d) < 4:
        return None, None, None, used_names, missing
    return (
        np.array(pts3d, dtype=np.float64),
        np.array(pts2d, dtype=np.float64),
        np.ones(len(pts2d), dtype=np.int32),
        used_names,
        missing,
    )


def solve_pnp(
    pts3d: np.ndarray,
    pts2d: np.ndarray,
    K: np.ndarray,
    method: str,
    ransac_thresh: float,
    dist: np.ndarray | None,
    refine: bool,
):
    # --- IPPE branch: planar専用 ---
    if method == "IPPE":
        zs = pts3d[:, 2]
        # ほぼ共面(z一定)でのみ IPPE を使う
        if np.allclose(zs, zs[0], atol=1e-9):
            out = cv2.solvePnPGeneric(
                objectPoints=pts3d.astype(np.float64),
                imagePoints=pts2d.astype(np.float64),
                cameraMatrix=K.astype(np.float64),
                distCoeffs=dist,
                flags=cv2.SOLVEPNP_IPPE,
                useExtrinsicGuess=False,
            )
            # OpenCV版差による返り値の長さに対応
            # 4要素: (retval, rvecs, tvecs, reprojErr)
            # 5要素: (retval, rvecs, tvecs, reprojErr, inliers) など
            if not isinstance(out, tuple | list) or len(out) < 4:
                return None, None, None

            retval = out[0]
            rvecs = out[1]
            tvecs = out[2]
            reproj = out[3]  # ない版もあるため下で吸収

            if not retval or rvecs is None or len(rvecs) == 0:
                # IPPEで解が出ない場合は下の一般PnPにフォールバック
                method = "EPNP"
            else:
                # 最良解の選択
                try:
                    err = np.asarray(reproj).reshape(-1)
                    idx = int(np.argmin(err)) if err.size > 0 else 0
                except Exception:
                    idx = 0
                rvec_sel = np.asarray(rvecs[idx], dtype=np.float64)
                tvec_sel = np.asarray(tvecs[idx], dtype=np.float64)
                # IPPEにはRANSACが無いので、全点をインライア扱いで返す
                return rvec_sel, tvec_sel, np.arange(len(pts2d))

        # ここに来たら IPPE不適合 or 失敗 → 一般PnPで続行
        method = "EPNP"

    # --- 一般PnP (RANSAC) ---
    flags_map = {
        "EPNP": cv2.SOLVEPNP_EPNP,
        "ITERATIVE": cv2.SOLVEPNP_ITERATIVE,
        "P3P": cv2.SOLVEPNP_P3P,
        "AP3P": cv2.SOLVEPNP_AP3P,
    }
    flag = flags_map[method]
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=pts3d.astype(np.float64),
        imagePoints=pts2d.astype(np.float64),
        cameraMatrix=K.astype(np.float64),
        distCoeffs=dist,
        reprojectionError=float(ransac_thresh),
        flags=flag,
    )
    if not ok:
        return None, None, None
    inlier_idx = np.arange(len(pts2d)) if inliers is None else inliers.flatten()

    if refine and len(inlier_idx) >= 4:
        ok2, rvec_ref, tvec_ref = cv2.solvePnP(
            objectPoints=pts3d[inlier_idx].astype(np.float64),
            imagePoints=pts2d[inlier_idx].astype(np.float64),
            cameraMatrix=K.astype(np.float64),
            distCoeffs=dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
            useExtrinsicGuess=True,
            rvec=rvec,
            tvec=tvec,
        )
        if ok2:
            rvec, tvec = rvec_ref, tvec_ref
    return rvec, tvec, inlier_idx


def project_points(pts3d: np.ndarray, K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, dist=None):
    pts2d_proj, _ = cv2.projectPoints(
        objectPoints=pts3d.astype(np.float64),
        rvec=rvec.astype(np.float64),
        tvec=tvec.astype(np.float64),
        cameraMatrix=K.astype(np.float64),
        distCoeffs=dist,
    )
    return pts2d_proj.reshape(-1, 2)


def draw_overlay(img_path: str, gt_pts: np.ndarray, prj_pts: np.ndarray, out_path: str):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        errer = f"Image not found: {img_path}"
        raise FileNotFoundError(errer)
    vis = img.copy()
    for (gx, gy), (px, py) in zip(gt_pts.astype(int), prj_pts.astype(int), strict=False):
        cv2.circle(vis, (int(gx), int(gy)), 4, (0, 255, 255), -1)
        cv2.circle(vis, (int(px), int(py)), 4, (0, 200, 0), -1)
        cv2.line(vis, (int(gx), int(gy)), (int(px), int(py)), (255, 0, 0), 1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)


def _mean_reproj_err(
    pts3d: np.ndarray, pts2d: np.ndarray, K: np.ndarray, rvec: np.ndarray | None, tvec: np.ndarray | None, dist
) -> float:
    if rvec is None or tvec is None:
        return float("inf")
    proj = project_points(pts3d, K, rvec, tvec, dist)
    return float(np.mean(np.linalg.norm(proj - pts2d, axis=1)))


def main():  # noqa: C901
    args = parse_args()
    np.random.seed(args.seed)

    images, anns_per_img, kp_names_coco = load_coco(args.ann)

    # court spec
    name2xyz = load_court_spec(args.court_spec)
    spec_keys = list(name2xyz.keys())

    # name resolution (now aware of spec keys)
    resolved_names = resolve_names(kp_names_coco, args.name_source, spec_keys)

    os.makedirs(args.out, exist_ok=True)
    overlay_dir = os.path.join(args.out, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    dist = parse_dist_coeffs(args.dist_coeffs)

    rows = []
    processed = 0
    total = len(anns_per_img)

    # Quick sanity: warn if none of resolved_names are present in spec
    if not any((n in name2xyz) for n in resolved_names):
        print(
            "[WARN] Resolved names do not match any entries in court_spec.yaml. "
            "Consider --name-source spec (or ensure spec keys match)."
        )

    for img_id, ann_list in anns_per_img.items():
        ann = ann_list[0]
        iminfo = images[img_id]
        width, height = int(iminfo["width"])

        cx = (width / 2.0) if args.cx == "auto" else float(args.cx)
        cy = (height / 2.0) if args.cy == "auto" else float(args.cy)
        K = Intrinsics(args.fx, args.fy, cx, cy).K()

        res = build_correspondences(ann, resolved_names, name2xyz)
        pts3d, pts2d, vis, used_names, missing = res
        if pts3d is None:
            if used_names is not None and missing is not None:
                print(
                    f"[INFO] Skip image {img_id}: insufficient correspondences (<4). "
                    f"Missing (first few): {list(set(missing))[:5]}"
                )
            else:
                print(f"[INFO] Skip image {img_id}: insufficient correspondences (<4).")
            continue

        # --- Solve (hypothesis 1: as-is) ---
        rvec1, tvec1, inlier_idx1 = solve_pnp(
            pts3d, pts2d, K, args.method, args.ransac_reproj_thresh, dist, args.refine
        )
        err1 = _mean_reproj_err(pts3d, pts2d, K, rvec1, tvec1, dist)

        # --- Solve (hypothesis 2: Y-mirrored) ---
        mirror_used = False
        rvec_sel, tvec_sel, inlier_idx_sel = rvec1, tvec1, inlier_idx1
        pts3d_sel = pts3d

        if args.try_mirror:
            pts3d_m = pts3d.copy()
            pts3d_m[:, 1] *= -1.0  # y -> -y
            rvec2, tvec2, inlier_idx2 = solve_pnp(
                pts3d_m, pts2d, K, args.method, args.ransac_reproj_thresh, dist, args.refine
            )
            err2 = _mean_reproj_err(pts3d_m, pts2d, K, rvec2, tvec2, dist)

            if err2 + 1e-9 < err1:
                mirror_used = True
                rvec_sel, tvec_sel, inlier_idx_sel = rvec2, tvec2, inlier_idx2
                pts3d_sel = pts3d_m

        if rvec_sel is None:
            print(f"[INFO] PnP failed on image {img_id}.")
            continue

        # Projection with the selected hypothesis
        proj = project_points(pts3d_sel, K, rvec_sel, tvec_sel, dist)
        diffs = proj - pts2d
        errs = np.linalg.norm(diffs, axis=1)
        inlier_mask = np.zeros_like(errs, dtype=bool)
        if inlier_idx_sel is not None:
            inlier_mask[inlier_idx_sel] = True
        mean_err = float(np.mean(errs))
        median_err = float(np.median(errs))
        mean_err_inliers = float(np.mean(errs[inlier_mask])) if np.any(inlier_mask) else mean_err

        rdeg = float(np.linalg.norm(np.rad2deg(rvec_sel.reshape(-1))))

        img_path = os.path.join(args.img_root, iminfo["file_name"])
        out_path = os.path.join(overlay_dir, f"{os.path.splitext(iminfo['file_name'])[0]}_overlay.png")
        try:
            draw_overlay(img_path, pts2d, proj, out_path)
        except Exception as e:
            print(f"[WARN] Overlay skipped for {iminfo['file_name']}: {e}")

        rows.append({
            "image_id": int(img_id),
            "file_name": iminfo["file_name"],
            "width": width,
            "height": height,
            "num_points": (len(pts2d)),
            "num_inliers": (len(inlier_idx_sel)) if inlier_idx_sel is not None else 0,
            "mean_err_px": mean_err,
            "median_err_px": median_err,
            "mean_err_inliers_px": mean_err_inliers,
            "rot_mag_deg": rdeg,
            "tvec_x_m": float(tvec_sel[0]),
            "tvec_y_m": float(tvec_sel[1]),
            "tvec_z_m": float(tvec_sel[2]),
            "method": args.method,
            "ransac_thresh": float(args.ransac_reproj_thresh),
            "dist_len": 0 if dist is None else (len(dist)),
            "name_source": args.name_source,
            "mirror_used": bool(mirror_used),
        })

        processed += 1
        if args.max_images > 0 and processed >= args.max_images:
            break

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out, "results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[PnP Demo] Processed {processed} / {total} images. Results -> {csv_path}")
    print(f"[PnP Demo] Overlays -> {overlay_dir}")


if __name__ == "__main__":
    main()
