from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf

from ..base import Bundle, Stage


class Export(Stage):
    """Save a minimal bundle summary (YAML) to results directory.

    Note: Require "report.eval_bundle" so this stage only runs
    after evaluation is complete, effectively making it the last stage.
    """

    def __init__(self, out_dir: str, save_yaml: bool = True, save_npz: bool = False):
        self.out_dir = out_dir
        self.save_yaml = bool(save_yaml)
        self.save_npz = bool(save_npz)

    # Ensure export runs after evaluation (and thus after upstream stages)
    required_inputs = ["report.eval_bundle"]
    produces = []
    STAGE_VERSION = "1.0.0"

    def run(self, B: Bundle) -> Bundle:
        # If bundle_id is present, save under subdir
        subdir = None
        if getattr(B, "bundle_id", None):
            subdir = str(B.bundle_id)
        out_dir = Path(self.out_dir) / (subdir or "")
        out_dir.mkdir(parents=True, exist_ok=True)

        def _round_list(arr, nd=6):
            import numpy as np

            if arr is None:
                return None
            a = np.asarray(arr, dtype=float)
            return np.round(a, nd).tolist()

        H_present = B.H is not None
        pose = None
        if B.pose_cw is not None:
            pose = {
                "R": _round_list(B.pose_cw.get("R")),
                "t": _round_list(B.pose_cw.get("t")),
                "method": B.pose_cw.get("method", ""),
            }

        # Prepare report with rounded residuals if present
        from copy import deepcopy

        report_out = deepcopy(B.report)
        if "refine_lm" in report_out:
            rl = report_out["refine_lm"]
            if rl.get("residuals") is not None:
                # Round to 3 decimals for compactness
                try:
                    import numpy as np

                    res_arr = np.asarray(rl["residuals"], dtype=float)
                    rl["residuals"] = np.round(res_arr, 3).tolist()
                except Exception:
                    pass
            if rl.get("inlier_mask") is not None:
                rl["inlier_mask"] = [int(x) for x in rl["inlier_mask"]]

        # Versions
        vers = B.report.get("versions", {})

        summary: Dict[str, Any] = {
            "K": B.K,
            "n_frames": len(B.frames),
            "n_court_points": len(B.court3d),
            "H_present": bool(H_present),
            "H": _round_list(B.H) if H_present else None,
            "pose_cw": pose,
            "report": report_out,
            # Phase 0: schema + pipeline + stage versions must be present
            "schema_version": vers.get("schema_version", {}),
            "pipeline_version": vers.get("pipeline_version", ""),
            "stage_versions": vers.get("stage_versions", []),
        }

        if self.save_yaml:
            OmegaConf.save(config=OmegaConf.create(summary), f=out_dir / "bundle.yaml")

        # Optional NPZ export
        if self.save_npz:
            import numpy as np

            npz = {
                "K": np.asarray(
                    [
                        [B.K.get("fx", 0.0), 0.0, B.K.get("cx", 0.0)],
                        [0.0, B.K.get("fy", 0.0), B.K.get("cy", 0.0)],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=float,
                )
                if B.K
                else None,
                "dist": np.asarray(
                    [
                        B.K.get("dist", {}).get("k1", 0.0),
                        B.K.get("dist", {}).get("k2", 0.0),
                        B.K.get("dist", {}).get("p1", 0.0),
                        B.K.get("dist", {}).get("p2", 0.0),
                        B.K.get("dist", {}).get("k3", 0.0),
                    ],
                    dtype=float,
                )
                if B.K
                else None,
            }
            # Flatten poses if present
            if B.poses:
                Rs = [p.get("R") for p in B.poses]
                ts = [p.get("t") for p in B.poses]
                npz["poses_R"] = np.asarray(Rs, dtype=float)
                npz["poses_t"] = np.asarray(ts, dtype=float)
            # Residuals from refine_lm if present
            if B.report.get("refine_lm", {}).get("residuals") is not None:
                npz["residuals_last"] = np.asarray(B.report["refine_lm"]["residuals"], dtype=float)
            if B.report.get("refine_lm", {}).get("inlier_mask") is not None:
                npz["inlier_mask_last"] = np.asarray(B.report["refine_lm"]["inlier_mask"], dtype=int)
            # Save
            np.savez(out_dir / "bundle.npz", **{k: v for k, v in npz.items() if v is not None})
        return B
