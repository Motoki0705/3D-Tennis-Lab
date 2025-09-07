from __future__ import annotations

from typing import Any, Dict, List, Tuple


# Schema and pipeline versioning
SCHEMA_VERSIONS: Dict[str, str] = {
    "frameobs": "frameobs@1.0.0",
    "bundle": "bundle@1.0.0",
    "result": "result@1.0.0",
}

PIPELINE_VERSION: str = "pnpkit@1.0.0"


def ensure_versions(B) -> None:
    """Ensure version stamps exist in B.report["versions"]."""
    vers = B.report.get("versions") or {}
    if "schema_version" not in vers:
        vers["schema_version"] = {
            "frameobs": SCHEMA_VERSIONS["frameobs"],
            "bundle": SCHEMA_VERSIONS["bundle"],
            "result": SCHEMA_VERSIONS["result"],
        }
    if "pipeline_version" not in vers:
        vers["pipeline_version"] = PIPELINE_VERSION
    if "stage_versions" not in vers:
        vers["stage_versions"] = []  # list of {name, version}
    B.report["versions"] = vers


def _has_attr_path(obj: Any, path: str) -> bool:
    # Supports dotted path under Bundle attributes, including B.report.*
    parts = path.split(".")
    cur: Any = obj
    for p in parts:
        if isinstance(cur, dict):
            if p not in cur:
                return False
            cur = cur[p]
        else:
            if not hasattr(cur, p):
                return False
            cur = getattr(cur, p)
    return cur is not None


def _fmt_cfg(cfg: Dict[str, Any]) -> str:
    try:
        import json

        return json.dumps(cfg, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(cfg)


def _validate_K(K: Dict[str, Any]) -> Tuple[bool, str]:
    if not K:
        return False, "K is missing"
    need = ["fx", "fy", "cx", "cy"]
    missing = [k for k in need if K.get(k) is None]
    if missing:
        return False, f"K missing fields: {missing}"
    return True, ""


def _visible_count_first(B) -> int:
    if not B.frames:
        return 0
    fr = B.frames[0]
    vis = 0
    for name, (u, v, vflag, w) in fr.keypoints.items():
        if vflag and (w or 0) > 0 and (B.court3d is None or name in B.court3d):
            vis += 1
    return vis


def _check_ref_integrity(B) -> List[str]:
    errs: List[str] = []
    if not B.frames or B.court3d is None:
        return errs
    names3d = set(B.court3d.keys())
    for fr in B.frames:
        bad = [k for k, tup in fr.keypoints.items() if (tup[2] and k not in names3d)]
        if bad:
            errs.append(f"Frame {fr.frame_idx}: {len(bad)} visible keypoints not in court3d: {sorted(bad)[:5]}")
    return errs


def can_run(stage, B) -> tuple[bool, list[str]]:
    """Non-raising readiness check. Returns (ok, missing_reasons)."""
    reqs = getattr(stage, "required_inputs", []) or []
    missing: List[str] = []
    advice: List[str] = []

    for r in reqs:
        if r == "frames":
            if not B.frames:
                missing.append("frames")
                advice.append("Provide FrameObs[] via I/O adapter or s00_load_inputs")
        elif r == "court3d":
            if not B.court3d or len(B.court3d) < 4:
                missing.append("court3d")
                advice.append("Load court_annotator spec or keypoints map with >=4 points")
        elif r == "K":
            ok, msg = _validate_K(B.K)
            if not ok:
                missing.append("K")
                advice.append(f"{msg}; set cameras.* or data.camera_index mapping")
        elif r == "pose_cw":
            if not B.pose_cw or B.pose_cw.get("R") is None or B.pose_cw.get("t") is None:
                missing.append("pose_cw")
                advice.append("Run s30_ippe_init or provide initial pose_cw")
        elif r == "min_visible_4":
            if _visible_count_first(B) < 4:
                missing.append("min_visible_4")
                advice.append("Need >=4 visible weighted points in first frame")
        elif r == "ref_integrity":
            errs = _check_ref_integrity(B)
            if errs:
                missing.append("ref_integrity")
                advice.extend(errs)
        else:
            if not _has_attr_path(B, r):
                missing.append(r)
                advice.append(f"Bundle missing attribute '{r}'")

    return (len(missing) == 0, advice)


def validate_required_inputs(stage, B) -> None:
    """Validate preconditions declared by stage.required_inputs.

    Supported tokens:
      - frames, court3d, K, pose_cw
      - min_visible_4: at least 4 visible weighted points in first frame intersecting court3d
      - ref_integrity: visible keypoints must be defined in court3d
    """
    ok, advice = can_run(stage, B)
    if not ok:
        name = getattr(stage, "STAGE_NAME", stage.__class__.__name__)
        raise ValueError(f"Stage '{name}' precondition failed. Guidance: " + "; ".join(advice))


def validate_produces(stage, B) -> None:
    prods = getattr(stage, "produces", []) or []
    missing: List[str] = []
    for p in prods:
        if not _has_attr_path(B, p):
            missing.append(p)
    if missing:
        name = getattr(stage, "STAGE_NAME", stage.__class__.__name__)
        raise ValueError(f"Stage '{name}' postcondition failed: did not produce {missing}")


def add_stage_version(stage, B) -> None:
    name = getattr(stage, "STAGE_NAME", stage.__class__.__name__)
    ver = getattr(stage, "STAGE_VERSION", "0.0.0")
    ensure_versions(B)
    sv = B.report["versions"].get("stage_versions") or []
    sv.append({"name": name, "version": ver})
    B.report["versions"]["stage_versions"] = sv
