from __future__ import annotations


import numpy as np

from ..base import Bundle, Stage
from ..base import register


def _k_inv_from_params(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[1.0 / fx, 0.0, -cx / fx], [0.0, 1.0 / fy, -cy / fy], [0.0, 0.0, 1.0]], dtype=np.float64)


@register("s20_estimate_k")
class EstimateK(Stage):
    required_inputs = ["H"]
    produces = ["K", "report.estimate_k"]
    STAGE_VERSION = "1.0.0"

    def __init__(
        self,
        principal_center: bool = True,
        assume_square_pixels: bool = True,
        enabled: bool = False,
        method: str = "vp_square",
        skip_if_K_present: bool = True,
        prior_weight: float = 0.05,
        f_bounds_scale: tuple[float, float] | None = (0.2, 3.0),
    ):
        # Persist params for caching/fingerprints
        super().__init__(
            principal_center=principal_center,
            assume_square_pixels=assume_square_pixels,
            enabled=enabled,
            method=method,
            skip_if_K_present=bool(skip_if_K_present),
            prior_weight=float(prior_weight),
            f_bounds_scale=list(f_bounds_scale) if f_bounds_scale is not None else None,
        )
        self.principal_center = principal_center
        self.assume_square_pixels = assume_square_pixels
        self.enabled = enabled
        self.method = method
        self.skip_if_K_present = bool(skip_if_K_present)
        self.prior_weight = float(prior_weight)
        self.f_bounds_scale = f_bounds_scale

    def should_skip(self, B: Bundle) -> bool:
        if not self.enabled:
            return True
        if self.skip_if_K_present and B.K is not None:
            if all(B.K.get(k) is not None for k in ("fx", "fy", "cx", "cy")):
                return True
        return False

    def run(self, B: Bundle) -> Bundle:
        if not self.enabled:
            return B
        if B.H is None:
            B.report["estimate_k"] = {"success": False, "reason": "H missing"}
            return B
        H = np.asarray(B.H, dtype=np.float64)
        vX = H[:, 0]
        vY = H[:, 1]

        # Determine starting cx, cy
        cx = float((B.K or {}).get("cx", 960.0))
        cy = float((B.K or {}).get("cy", 540.0))

        # Estimate a reasonable image diagonal for bounds
        diag = None
        try:
            if B.frames and B.frames[0].meta:
                W = float(B.frames[0].meta.get("width", 2 * cx))
                Hh = float(B.frames[0].meta.get("height", 2 * cy))
                diag = float(np.hypot(W, Hh))
            else:
                diag = float(np.hypot(2 * cx, 2 * cy))
        except Exception:
            diag = None

        # Guard against degenerate vanishing points
        if abs(vX[2]) < 1e-9 or abs(vY[2]) < 1e-9:
            B.report["estimate_k"] = {"success": False, "reason": "degenerate vanishing points"}
            return B

        # Normalize homogeneous to pixel space for stability (scale-invariant with K^{-1})
        vXn = vX / vX[2]
        vYn = vY / vY[2]

        def residuals_from_params(fx: float, fy: float, cx0: float, cy0: float) -> np.ndarray:
            Kinv = _k_inv_from_params(fx, fy, cx0, cy0)
            a = Kinv @ vXn
            b = Kinv @ vYn
            r1 = a[:3].dot(b[:3])
            r2 = a[:3].dot(a[:3]) - b[:3].dot(b[:3])
            return np.array([r1, r2], dtype=np.float64)

        # Dynamic bounds based on image diagonal if available
        fmin, fmax = 200.0, 10000.0
        if diag is not None and self.f_bounds_scale is not None:
            lo_s, hi_s = self.f_bounds_scale
            fmin = max(fmin, float(lo_s) * diag)
            fmax = min(fmax, float(hi_s) * diag)
        success = False
        fx_est, fy_est, cx_est, cy_est = None, None, None, None
        if self.principal_center and self.assume_square_pixels:
            # 1D search over f
            def cost(f: float) -> float:
                r = residuals_from_params(f, f, cx, cy)
                # weak prior to avoid runaway f
                f0 = float(((B.K or {}).get("fx", 1200.0)) * ((B.K or {}).get("fy", 1200.0))) ** 0.5
                prior = ((f / max(1e-6, f0)) - 1.0) ** 2
                return float(r @ r) + self.prior_weight * prior

            fs = np.geomspace(fmin, fmax, num=64)
            costs = [cost(float(f)) for f in fs]
            f_best = float(fs[int(np.argmin(costs))])
            # Local refine by simple ternary-like shrink
            left = max(fmin, f_best / 2)
            right = min(fmax, f_best * 2)
            for _ in range(30):
                f1 = left + (right - left) / 3
                f2 = right - (right - left) / 3
                c1, c2 = cost(f1), cost(f2)
                if c1 < c2:
                    right = f2
                else:
                    left = f1
            f_opt = (left + right) * 0.5
            fx_est = float(np.clip(f_opt, fmin, fmax))
            fy_est = fx_est
            cx_est, cy_est = cx, cy
            success = True
        else:
            # Use SciPy if available
            try:
                from scipy.optimize import least_squares
            except Exception:
                B.report["estimate_k"] = {"success": False, "reason": "scipy missing"}
                return B

            vars0 = []
            names = []
            if True:
                # fx,fy always estimated in non-square case
                vars0 += [float((B.K or {}).get("fx", 1200.0)), float((B.K or {}).get("fy", 1200.0))]
                names += ["fx", "fy"]
            if not self.principal_center:
                vars0 += [cx, cy]
                names += ["cx", "cy"]

            lb = []
            ub = []
            for nm in names:
                if nm in ("fx", "fy"):
                    lb.append(fmin)
                    ub.append(fmax)
                elif nm == "cx":
                    lb.append(0.0)
                    ub.append(4096.0)
                elif nm == "cy":
                    lb.append(0.0)
                    ub.append(4096.0)

            def res_fn(x):
                vals = {k: v for k, v in zip(names, x)}
                fxv = float(vals.get("fx", (B.K or {}).get("fx", 1200.0)))
                fyv = float(vals.get("fy", (B.K or {}).get("fy", 1200.0)))
                cxv = float(vals.get("cx", cx))
                cyv = float(vals.get("cy", cy))
                return residuals_from_params(fxv, fyv, cxv, cyv)

            ls = least_squares(res_fn, np.asarray(vars0, dtype=np.float64), bounds=(np.asarray(lb), np.asarray(ub)))
            vals = {k: v for k, v in zip(names, ls.x)}
            fx_est = float(vals.get("fx", (B.K or {}).get("fx", 1200.0)))
            fy_est = float(vals.get("fy", (B.K or {}).get("fy", 1200.0)))
            cx_est = float(vals.get("cx", cx))
            cy_est = float(vals.get("cy", cy))
            success = ls.success

        if success:
            # sanity range check: reject clearly implausible focal lengths
            out_of_range = False
            if diag is not None:
                lo_ok = 0.3 * diag
                hi_ok = 3.0 * diag
                if not (lo_ok <= fx_est <= hi_ok and lo_ok <= fy_est <= hi_ok):
                    out_of_range = True

            if out_of_range:
                B.report["estimate_k"] = {
                    "method": self.method,
                    "success": False,
                    "fx": float(fx_est),
                    "fy": float(fy_est),
                    "cx": float(cx_est),
                    "cy": float(cy_est),
                    "reason": "out_of_range",
                }
                return B

            B.K = B.K or {}
            B.K.update({
                "fx": fx_est,
                "fy": fy_est,
                "cx": cx_est,
                "cy": cy_est,
                "skew": float((B.K or {}).get("skew", 0.0)),
            })
        B.report["estimate_k"] = {
            "method": self.method,
            "success": bool(success),
            "fx": float(fx_est) if fx_est is not None else None,
            "fy": float(fy_est) if fy_est is not None else None,
            "cx": float(cx_est) if cx_est is not None else None,
            "cy": float(cy_est) if cy_est is not None else None,
        }
        return B
