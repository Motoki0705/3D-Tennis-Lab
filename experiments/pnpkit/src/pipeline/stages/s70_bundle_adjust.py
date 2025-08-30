from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import cv2

from ..base import Bundle, Stage
from ..base import register
from ...core.camera import Intrinsics, PoseCW
from ...core.geometry import project_points_W


def _gather_frame_matches(court3d: Dict[str, np.ndarray], keypoints: Dict[str, Tuple[float, float, int, float]]):
    Xw: List[List[float]] = []
    U: List[List[float]] = []
    for name, xyz in court3d.items():
        if name not in keypoints:
            continue
        u, v, vis, w = keypoints[name]
        if vis and w > 0:
            Xw.append([float(xyz[0]), float(xyz[1]), 0.0])
            U.append([float(u), float(v)])
    return np.asarray(Xw, dtype=np.float64), np.asarray(U, dtype=np.float64)


@register("s70_bundle_adjust")
class BundleAdjust(Stage):
    required_inputs = ["frames", "court3d", "K"]
    produces = ["poses", "K", "report.bundle_adjust"]
    STAGE_VERSION = "1.0.0"

    def run(self, B: Bundle) -> Bundle:
        try:
            from scipy.optimize import least_squares
        except Exception as e:
            raise ImportError("BundleAdjust requires SciPy (scipy.optimize.least_squares)") from e

        if not B.frames or len(B.frames) < 1:
            raise ValueError("BundleAdjust: need >=1 frame")
        if not B.K:
            raise ValueError("BundleAdjust: Bundle.K missing")

        # Intrinsics initial
        Kd = B.K
        K_vars = {
            "fx": float(Kd.get("fx", 1200.0)),
            "fy": float(Kd.get("fy", 1200.0)),
            "cx": float(Kd.get("cx", 960.0)),
            "cy": float(Kd.get("cy", 540.0)),
        }
        D_vars = {"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0}
        if Kd.get("dist"):
            d = Kd["dist"]
            D_vars.update({
                "k1": float(d.get("k1", 0.0)),
                "k2": float(d.get("k2", 0.0)),
                "p1": float(d.get("p1", 0.0)),
                "p2": float(d.get("p2", 0.0)),
                "k3": float(d.get("k3", 0.0)),
            })

        # Poses initial: from B.poses if present else from B.pose_cw (first) and identity for others
        poses: List[Dict[str, Any]] = []
        if B.poses:
            poses = [
                {
                    "frame_idx": p.get("frame_idx", i),
                    "R": np.asarray(p["R"], dtype=np.float64),
                    "t": np.asarray(p["t"], dtype=np.float64).reshape(3),
                }
                for i, p in enumerate(B.poses)
            ]
        else:
            poses = []
            for i, fr in enumerate(B.frames):
                if i == 0 and B.pose_cw is not None:
                    poses.append({
                        "frame_idx": fr.frame_idx,
                        "R": np.asarray(B.pose_cw["R"], dtype=np.float64),
                        "t": np.asarray(B.pose_cw["t"], dtype=np.float64).reshape(3),
                    })
                else:
                    poses.append({
                        "frame_idx": fr.frame_idx,
                        "R": np.eye(3, dtype=np.float64),
                        "t": np.array([0.0, 0.0, 10.0], dtype=np.float64),
                    })

        # Build per-frame matches
        frames_X = []
        frames_uv = []
        for fr in B.frames:
            Xw, uv = _gather_frame_matches(B.court3d, fr.keypoints)
            frames_X.append(Xw)
            frames_uv.append(uv)

        # Config
        refine_flags = {
            k: bool(v)
            for k, v in (self.cfg.get("refine", {}) or {}).items()
            if k in {"R", "t", "fx", "fy", "cx", "cy", "dist"}
        }
        schedule = self.cfg.get("schedule", []) or [{"unlock": [k for k, v in refine_flags.items() if v]}]
        robust = self.cfg.get("robust", {}) or {}
        loss = robust.get("loss", "huber")
        f_scale = float(robust.get("f_scale", 1.0))
        bcfg = self.cfg.get("bounds", {}) or {}
        b_fx = tuple(bcfg.get("fx", [200.0, 10000.0]))
        b_fy = tuple(bcfg.get("fy", [200.0, 10000.0]))
        b_cx = tuple(bcfg.get("cx", [0.0, 4096.0]))
        b_cy = tuple(bcfg.get("cy", [0.0, 4096.0]))
        b_k = tuple(bcfg.get("k", [-0.5, 0.5]))
        b_p = tuple(bcfg.get("p", [-0.1, 0.1]))

        def build_intr(Kv, Dv) -> Intrinsics:
            intr = Intrinsics(
                fx=Kv["fx"], fy=Kv["fy"], cx=Kv["cx"], cy=Kv["cy"], skew=float(Kd.get("skew", 0.0) or 0.0)
            )
            intr.dist.k1 = Dv["k1"]
            intr.dist.k2 = Dv["k2"]
            intr.dist.p1 = Dv["p1"]
            intr.dist.p2 = Dv["p2"]
            intr.dist.k3 = Dv["k3"]
            return intr

        def flatten_vars(enabled: List[str]) -> np.ndarray:
            v = []
            # per-frame R,t
            for p in poses:
                if "R" in enabled:
                    rvec, _ = cv2.Rodrigues(p["R"])  # type: ignore
                    v.extend(rvec.reshape(3).tolist())
                if "t" in enabled:
                    v.extend(p["t"].reshape(3).tolist())
            if "fx" in enabled:
                v.append(K_vars["fx"])
            if "fy" in enabled:
                v.append(K_vars["fy"])
            if "cx" in enabled:
                v.append(K_vars["cx"])
            if "cy" in enabled:
                v.append(K_vars["cy"])
            if "dist" in enabled:
                v.extend([D_vars[k] for k in ("k1", "k2", "p1", "p2", "k3")])
            return np.asarray(v, dtype=np.float64)

        def unflatten_vars(x: np.ndarray, enabled: List[str]):
            i = 0
            new_poses = []
            for p in poses:
                R = p["R"]
                t = p["t"]
                if "R" in enabled:
                    rvec = x[i : i + 3]
                    i += 3
                    R, _ = cv2.Rodrigues(rvec)
                if "t" in enabled:
                    t = x[i : i + 3]
                    i += 3
                new_poses.append({"frame_idx": p["frame_idx"], "R": R, "t": t})
            Kv = K_vars.copy()
            Dv = D_vars.copy()
            if "fx" in enabled:
                Kv["fx"] = float(x[i])
                i += 1
            if "fy" in enabled:
                Kv["fy"] = float(x[i])
                i += 1
            if "cx" in enabled:
                Kv["cx"] = float(x[i])
                i += 1
            if "cy" in enabled:
                Kv["cy"] = float(x[i])
                i += 1
            if "dist" in enabled:
                for k in ("k1", "k2", "p1", "p2", "k3"):
                    Dv[k] = float(x[i])
                    i += 1
            return new_poses, Kv, Dv

        def build_bounds(enabled: List[str]):
            lb = []
            ub = []
            for _ in poses:
                if "R" in enabled:
                    lb += [-np.inf, -np.inf, -np.inf]
                    ub += [np.inf, np.inf, np.inf]
                if "t" in enabled:
                    lb += [-np.inf, -np.inf, -np.inf]
                    ub += [np.inf, np.inf, np.inf]
            if "fx" in enabled:
                lb += [b_fx[0]]
                ub += [b_fx[1]]
            if "fy" in enabled:
                lb += [b_fy[0]]
                ub += [b_fy[1]]
            if "cx" in enabled:
                lb += [b_cx[0]]
                ub += [b_cx[1]]
            if "cy" in enabled:
                lb += [b_cy[0]]
                ub += [b_cy[1]]
            if "dist" in enabled:
                lb += [b_k[0], b_k[0], b_p[0], b_p[0], b_k[0]]
                ub += [b_k[1], b_k[1], b_p[1], b_p[1], b_k[1]]
            return np.asarray(lb, dtype=np.float64), np.asarray(ub, dtype=np.float64)

        # Compute initial RMSE
        intr0 = build_intr(K_vars, D_vars)
        sq = []

        for p, Xw, uv in zip(poses, frames_X, frames_uv):
            uvh = project_points_W(Xw, intr0, PoseCW(R=p["R"], t=p["t"]))
            sq.append(np.sum((uvh - uv) ** 2))
        rmse_before = float(np.sqrt(np.sum(sq) / max(1, sum(len(u) for u in frames_uv))))

        passes = []
        last_residuals = None
        for step in schedule:
            unlock = step.get("unlock", [])
            if isinstance(unlock, str):
                unlock = [unlock]
            enabled = []
            if refine_flags.get("R", True):
                enabled.append("R")
            if refine_flags.get("t", True):
                enabled.append("t")
            for name in unlock:
                if name in ("R", "t"):
                    continue
                if name in ("fx", "fy", "cx", "cy", "dist") and refine_flags.get(
                    name if name != "dist" else "dist", False
                ):
                    if name not in enabled:
                        enabled.append(name)

            x0 = flatten_vars(enabled)
            lb, ub = build_bounds(enabled)

            def residual_fn(x):
                new_poses, Kv, Dv = unflatten_vars(x, enabled)
                intr_l = build_intr(Kv, Dv)
                res_list = []
                for p, Xw, uv in zip(new_poses, frames_X, frames_uv):
                    uv_hat = project_points_W(Xw, intr_l, PoseCW(R=p["R"], t=p["t"]))
                    res_list.append((uv_hat - uv).reshape(-1))
                return np.concatenate(res_list, axis=0)

            ls = least_squares(
                residual_fn, x0, bounds=(lb, ub), loss=loss, f_scale=f_scale, max_nfev=int(step.get("max_nfev", 100))
            )
            poses, K_vars, D_vars = unflatten_vars(ls.x, enabled)
            intr_cur = build_intr(K_vars, D_vars)
            res_list = []
            for p, Xw, uv in zip(poses, frames_X, frames_uv):
                uv_hat = project_points_W(Xw, intr_cur, PoseCW(R=p["R"], t=p["t"]))
                res_list.append((uv_hat - uv).reshape(-1))
            last_residuals = np.concatenate(res_list, axis=0)
            rmse_cur = float(np.sqrt(np.mean(last_residuals**2)))
            passes.append({"enabled": enabled, "rmse_px": rmse_cur, "nfev": int(ls.nfev)})

        # Finalize
        B.K.update({"fx": K_vars["fx"], "fy": K_vars["fy"], "cx": K_vars["cx"], "cy": K_vars["cy"]})
        B.K["dist"] = {
            "k1": D_vars["k1"],
            "k2": D_vars["k2"],
            "p1": D_vars["p1"],
            "p2": D_vars["p2"],
            "k3": D_vars["k3"],
        }
        B.poses = poses
        # Also set first pose for compatibility
        if poses:
            B.pose_cw = {"R": poses[0]["R"], "t": poses[0]["t"], "method": "BundleAdjust"}

        total_pts = int(sum(len(u) for u in frames_uv))
        rmse_after = float(passes[-1]["rmse_px"]) if passes else rmse_before
        B.report.setdefault("bundle_adjust", {})
        B.report["bundle_adjust"].update({
            "rmse_px_before": rmse_before,
            "rmse_px_after": rmse_after,
            "n_points_total": total_pts,
            "n_frames": len(B.frames),
            "passes": passes,
        })
        return B
