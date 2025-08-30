from __future__ import annotations

from typing import Dict, Tuple, List, Any

import numpy as np
import cv2

from ..base import Bundle, Stage
from ..base import register
from ...core.camera import Intrinsics, PoseCW
from ...core.geometry import project_points_W, invert_pose


def _gather_matches3d2d(court3d: Dict[str, np.ndarray], keypoints: Dict[str, Tuple[float, float, int, float]]):
    Xw: List[List[float]] = []
    U: List[List[float]] = []
    for name, xyz in court3d.items():
        if name not in keypoints:
            continue
        u, v, vis, w = keypoints[name]
        if vis and w > 0:
            Xw.append([float(xyz[0]), float(xyz[1]), 0.0])
            U.append([float(u), float(v)])
    Xw = np.asarray(Xw, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    if len(U) < 4:
        raise ValueError(f"RefineLM: need >=4 visible weighted points, got {len(U)}")
    return Xw, U


@register("s40_refine_lm")
class RefineLM(Stage):
    required_inputs = ["frames", "court3d", "K", "pose_cw", "min_visible_4", "ref_integrity"]
    produces = ["pose_cw", "report.refine_lm"]
    STAGE_VERSION = "1.0.0"

    def run(self, B: Bundle) -> Bundle:
        if not B.frames:
            raise ValueError("RefineLM: Bundle.frames is empty")
        if not B.K:
            raise ValueError("RefineLM: Bundle.K is missing (need fx, fy, cx, cy)")
        if not B.pose_cw:
            raise ValueError("RefineLM: initial pose_cw missing; run IPPEInit first")

        kp = B.frames[0].keypoints
        Xw, uv = _gather_matches3d2d(B.court3d, kp)

        Kd = B.K
        fx, fy, cx, cy = Kd.get("fx"), Kd.get("fy"), Kd.get("cx"), Kd.get("cy")
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        dist = None
        if "dist" in Kd and Kd["dist"] is not None:
            d = Kd["dist"]
            dist = np.array(
                [
                    float(d.get("k1", 0.0)),
                    float(d.get("k2", 0.0)),
                    float(d.get("p1", 0.0)),
                    float(d.get("p2", 0.0)),
                    float(d.get("k3", 0.0)),
                ],
                dtype=np.float64,
            )

        # Initial pose
        R0 = np.asarray(B.pose_cw.get("R"), dtype=np.float64)
        t0 = np.asarray(B.pose_cw.get("t"), dtype=np.float64).reshape(3)
        rvec0, _ = cv2.Rodrigues(R0)
        tvec0 = t0.reshape(3, 1)

        # Compute initial residuals and inliers
        intr = Intrinsics(
            fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy), skew=float(Kd.get("skew", 0.0) or 0.0)
        )
        if "dist" in Kd and Kd["dist"] is not None:
            d = Kd["dist"]
            intr.dist.k1 = float(d.get("k1", 0.0))
            intr.dist.k2 = float(d.get("k2", 0.0))
            intr.dist.p1 = float(d.get("p1", 0.0))
            intr.dist.p2 = float(d.get("p2", 0.0))
            intr.dist.k3 = float(d.get("k3", 0.0))
        uv_hat0 = project_points_W(Xw, intr, PoseCW(R=R0, t=t0))
        res0 = uv_hat0 - uv
        rmse_before = float(np.sqrt(np.mean(np.sum(res0**2, axis=1))))

        thr = float(self.cfg.get("inlier_threshold_px", 3.0))
        inliers = np.sum(np.sum(res0**2, axis=1) <= (thr * thr))

        # Use only inliers if available
        if inliers >= 4:
            mask = np.sum(res0**2, axis=1) <= (thr * thr)
            Xw_ref = Xw[mask]
            uv_ref = uv[mask]
        else:
            Xw_ref = Xw
            uv_ref = uv

        backend = (self.cfg.get("refine", {}) or {}).get("backend", "opencv")
        if backend == "opencv":
            # Refine pose using OpenCV LM (pose-only)
            rvec_ref, tvec_ref = cv2.solvePnPRefineLM(
                Xw_ref,
                uv_ref,
                K,
                dist if dist is not None else np.zeros(5, dtype=np.float64),
                rvec0.astype(np.float64),
                tvec0.astype(np.float64),
            )
            R_ref, _ = cv2.Rodrigues(rvec_ref)
            t_ref = tvec_ref.reshape(3)
            uv_hat = project_points_W(Xw, intr, PoseCW(R=R_ref, t=t_ref))
            res = uv_hat - uv
            rmse_after = float(np.sqrt(np.mean(np.sum(res**2, axis=1))))

            B.pose_cw = {"R": R_ref, "t": t_ref, "method": "RefineLM(opencv)"}
            B.report.setdefault("refine_lm", {})
            B.report["refine_lm"].update({
                "rmse_px_before": rmse_before,
                "rmse_px_after": rmse_after,
                "n_inliers": int(inliers),
                "total": int(len(uv)),
                "threshold_px": thr,
                "backend": "opencv",
            })
            # Store last residuals and inlier mask for export
            B.report["refine_lm"]["residuals"] = res.tolist()
            B.report["refine_lm"]["inlier_mask"] = (np.sum(res**2, axis=1) <= thr * thr).astype(int).tolist()
            return B

        elif backend == "scipy":
            try:
                from scipy.optimize import least_squares
            except Exception as e:
                raise ImportError("RefineLM(scipy): SciPy is required for scipy backend") from e

            # Global flags and schedule
            refine_flags: Dict[str, bool] = {
                k: bool(v)
                for k, v in (self.cfg.get("refine", {}) | {}).items()
                if k in {"R", "t", "fx", "fy", "cx", "cy", "dist"}
            }
            schedule = self.cfg.get("schedule", [])
            robust = self.cfg.get("robust", {}) or {}
            loss = robust.get("loss", "huber")
            f_scale = float(robust.get("f_scale", 1.0))

            # Initial variables
            rvec_init = rvec0.reshape(3).copy()
            t_init = t0.reshape(3).copy()
            K_vars = {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}
            D_vars = {"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0}
            if dist is not None:
                vals = dist.reshape(-1).tolist() + [0.0] * (5 - len(dist))
                D_vars = {"k1": vals[0], "k2": vals[1], "p1": vals[2], "p2": vals[3], "k3": vals[4]}

            # Bounds
            bcfg = self.cfg.get("bounds", {}) or {}
            b_fx = tuple(bcfg.get("fx", [200.0, 10000.0]))
            b_fy = tuple(bcfg.get("fy", [200.0, 10000.0]))
            b_cx = tuple(bcfg.get("cx", [0.0, 4096.0]))
            b_cy = tuple(bcfg.get("cy", [0.0, 4096.0]))
            b_k = tuple(bcfg.get("k", [-0.5, 0.5]))
            b_p = tuple(bcfg.get("p", [-0.1, 0.1]))

            # Priors
            pri = self.cfg.get("priors", {}) or {}
            pri_use = bool(pri.get("use", False))
            pri_w = {"height": 1.0, "roll": 1.0, "pitch": 1.0, "fx": 0.1} | (pri.get("weight", {}) or {})
            height_mu = float((pri.get("height_m", {}) or {}).get("mu", 8.0))
            height_sig = float((pri.get("height_m", {}) or {}).get("sigma", 4.0))
            roll_mu = float((pri.get("roll_deg", {}) or {}).get("mu", 0.0))
            roll_sig = float((pri.get("roll_deg", {}) or {}).get("sigma", 10.0))
            pitch_mu = float((pri.get("pitch_deg", {}) or {}).get("mu", 0.0))
            pitch_sig = float((pri.get("pitch_deg", {}) or {}).get("sigma", 15.0))
            fx_rel_mu = float((pri.get("fx_rel", {}) or {}).get("mu_mult", 1.0))
            fx_rel_sig = float((pri.get("fx_rel", {}) or {}).get("sigma_frac", 0.5))
            fx0_ref = float(fx)

            def build_intrinsics(K_vars_local: Dict[str, float], D_vars_local: Dict[str, float]) -> Intrinsics:
                intr_l = Intrinsics(
                    fx=K_vars_local["fx"],
                    fy=K_vars_local["fy"],
                    cx=K_vars_local["cx"],
                    cy=K_vars_local["cy"],
                    skew=float(Kd.get("skew", 0.0) or 0.0),
                )
                intr_l.dist.k1 = D_vars_local["k1"]
                intr_l.dist.k2 = D_vars_local["k2"]
                intr_l.dist.p1 = D_vars_local["p1"]
                intr_l.dist.p2 = D_vars_local["p2"]
                intr_l.dist.k3 = D_vars_local["k3"]
                return intr_l

            def pack_vars(enabled: List[str]) -> np.ndarray:
                v = []
                if "R" in enabled:
                    v.extend(rvec_init.tolist())
                if "t" in enabled:
                    v.extend(t_init.tolist())
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

            def unpack_vars(x: np.ndarray, enabled: List[str]):
                i = 0
                if "R" in enabled:
                    nonlocal_r = x[i : i + 3]
                    i += 3
                else:
                    nonlocal_r = rvec_init
                if "t" in enabled:
                    nonlocal_t = x[i : i + 3]
                    i += 3
                else:
                    nonlocal_t = t_init
                Rv = np.asarray(nonlocal_r, dtype=np.float64)
                Tv = np.asarray(nonlocal_t, dtype=np.float64)
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
                return Rv, Tv, Kv, Dv

            def build_bounds(enabled: List[str]):
                lb = []
                ub = []
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

            rmse_before_scipy = rmse_before
            last_residuals = None
            last_mask = None
            passes_done: List[Dict[str, Any]] = []

            # Determine enabled sets per pass
            if not schedule:
                schedule = [{"unlock": [k for k, v in refine_flags.items() if v]}]
            for step in schedule:
                unlock = step.get("unlock", [])
                if isinstance(unlock, str):
                    unlock = [unlock]
                # Always include R/t if globally enabled
                enabled = []
                if refine_flags.get("R", True):
                    enabled.append("R")
                if refine_flags.get("t", True):
                    enabled.append("t")
                for name in unlock:
                    if name in ("R", "t"):
                        # already added above if enabled globally
                        continue
                    if name in ("fx", "fy", "cx", "cy", "dist") and refine_flags.get(
                        name if name != "dist" else "dist", False
                    ):
                        if name not in enabled:
                            enabled.append(name)

                x0 = pack_vars(enabled)
                lb, ub = build_bounds(enabled)

                def residual_fn(x):
                    Rv, Tv, Kv, Dv = unpack_vars(x, enabled)
                    intr_l = build_intrinsics(Kv, Dv)
                    uv_hat = project_points_W(Xw, intr_l, PoseCW(R=cv2.Rodrigues(Rv)[0], t=Tv))
                    res = (uv_hat - uv).reshape(-1)
                    # Priors
                    if pri_use:
                        # camera height
                        Rmat = cv2.Rodrigues(Rv)[0]
                        Rwc, twc = invert_pose(PoseCW(R=Rmat, t=Tv))
                        h = twc[2]
                        res = np.concatenate([
                            res,
                            pri_w["height"] * (np.array([(h - height_mu) / max(height_sig, 1e-6)], dtype=np.float64)),
                        ])
                        # roll/pitch from Rwc (XYZ intrinsic -> roll (x), pitch (y))
                        # Using standard Tait–Bryan ZYX: roll = atan2(R32,R33), pitch = -asin(R31)
                        roll = np.degrees(np.arctan2(Rwc[2, 1], Rwc[2, 2]))
                        pitch = -np.degrees(np.arcsin(np.clip(Rwc[2, 0], -1.0, 1.0)))
                        res = np.concatenate([
                            res,
                            pri_w["roll"] * np.array([(roll - roll_mu) / max(roll_sig, 1e-6)], dtype=np.float64),
                            pri_w["pitch"] * np.array([(pitch - pitch_mu) / max(pitch_sig, 1e-6)], dtype=np.float64),
                        ])
                        # fx relative prior
                        res = np.concatenate([
                            res,
                            pri_w["fx"]
                            * np.array(
                                [(Kv["fx"] / max(fx0_ref, 1e-9) - fx_rel_mu) / max(fx_rel_sig, 1e-6)], dtype=np.float64
                            ),
                        ])
                    return res

                max_nfev = int(step.get("max_nfev", 100))
                ls = least_squares(
                    residual_fn,
                    x0,
                    bounds=(lb, ub),
                    loss=loss,
                    f_scale=f_scale,
                    max_nfev=max_nfev,
                )

                # Update variables for next passes
                Rv, Tv, Kv, Dv = unpack_vars(ls.x, enabled)
                rvec_init = Rv
                t_init = Tv
                K_vars.update(Kv)
                D_vars.update(Dv)

                intr_cur = build_intrinsics(K_vars, D_vars)
                uv_hat_cur = project_points_W(Xw, intr_cur, PoseCW(R=cv2.Rodrigues(rvec_init)[0], t=t_init))
                res_cur = uv_hat_cur - uv
                rmse_cur = float(np.sqrt(np.mean(np.sum(res_cur**2, axis=1))))
                last_residuals = res_cur
                last_mask = (np.sum(res_cur**2, axis=1) <= thr * thr).astype(int)

                passes_done.append({"enabled": enabled, "rmse_px": rmse_cur, "nfev": int(ls.nfev)})

            # Finalize
            R_final = cv2.Rodrigues(rvec_init)[0]
            t_final = t_init.reshape(3)
            B.pose_cw = {"R": R_final, "t": t_final, "method": "RefineLM(scipy)"}
            # Update K back to bundle
            B.K.update({
                "fx": K_vars["fx"],
                "fy": K_vars["fy"],
                "cx": K_vars["cx"],
                "cy": K_vars["cy"],
                "skew": float(Kd.get("skew", 0.0) or 0.0),
            })
            B.K["dist"] = {
                "k1": D_vars["k1"],
                "k2": D_vars["k2"],
                "p1": D_vars["p1"],
                "p2": D_vars["p2"],
                "k3": D_vars["k3"],
            }

            B.report.setdefault("refine_lm", {})
            B.report["refine_lm"].update({
                "rmse_px_before": rmse_before_scipy,
                "rmse_px_after": float(passes_done[-1]["rmse_px"]) if passes_done else rmse_before_scipy,
                "n_inliers": int(np.sum(last_mask)) if last_mask is not None else int(inliers),
                "total": int(len(uv)),
                "threshold_px": thr,
                "backend": "scipy",
                "passes": passes_done,
                "residuals": last_residuals.tolist() if last_residuals is not None else None,
                "inlier_mask": last_mask.tolist() if last_mask is not None else None,
            })
            return B
        else:
            raise ValueError(f"RefineLM: unknown backend '{backend}'")
