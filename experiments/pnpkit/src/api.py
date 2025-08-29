from __future__ import annotations

from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from omegaconf import OmegaConf

from .pipeline.base import Bundle
from .pipeline.bundler import group_frames
from .utils.hydra_tools import instantiate_stages
from .pipeline.executor import run_dag
from .io.reader import adapt_frames, load_camera_index
from .io.court import court3d_from_cfg
from .core.camera import Intrinsics


def _load_frames(data_cfg: Dict[str, Any]):
    frames, warnings = adapt_frames(data_cfg)
    return frames, warnings


def run_bundled(
    data_cfg: Dict[str, Any],
    bundling_cfg: Dict[str, Any],
    pipeline_stage_cfgs: List[Dict[str, Any]],
    courts_cfg: Dict[str, Any],
    cameras_cfg: Dict[str, Any],
    parallel: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run I/O -> Bundler -> Pipeline for all bundles.

    Returns a mapping: bundle_id -> {"report": dict, "out_dir": str}
    """
    frames, global_warnings = _load_frames(data_cfg)
    strategy = bundling_cfg.get("strategy", "by_prefix")
    params = bundling_cfg.get("params", {})
    grouped = group_frames(frames, strategy=strategy, **params)

    court3d = court3d_from_cfg(OmegaConf.to_container(courts_cfg, resolve=True))
    K_default = cameras_cfg
    mapping = None
    if data_cfg.get("camera_index"):
        mapping = load_camera_index(data_cfg["camera_index"])
    gran = str((data_cfg.get("cameras_granularity") or "by_bundle"))

    # Build stages once
    stages = instantiate_stages(pipeline_stage_cfgs)

    def _process(bundle_id: str, frs):
        # choose intrinsics per bundle (prefix)
        K_cfg = K_default
        if gran == "global":
            K_cfg = K_default
        elif mapping and frs:
            import os

            bn = os.path.basename(frs[0].image_path)
            for pref, intr_path in mapping.items():
                if bn.startswith(pref):
                    K_cfg = OmegaConf.load(intr_path)
                    break
        Kintr = Intrinsics.from_dict(OmegaConf.to_container(K_cfg, resolve=True))
        B = Bundle(bundle_id=bundle_id, court3d=court3d, frames=frs)
        B.K = {
            "fx": Kintr.fx,
            "fy": Kintr.fy,
            "cx": Kintr.cx,
            "cy": Kintr.cy,
            "skew": Kintr.skew,
            "dist": {
                "k1": Kintr.dist.k1,
                "k2": Kintr.dist.k2,
                "p1": Kintr.dist.p1,
                "p2": Kintr.dist.p2,
                "k3": Kintr.dist.k3,
            },
        }
        # attach I/O warnings
        if global_warnings:
            B.report.setdefault("warnings", {})
            B.report["warnings"].setdefault("io", []).extend(global_warnings)

        # Run via DAG engine
        B = run_dag(
            B,
            stages,
            parallel=bool((parallel or {}).get("enable", False)),
            max_workers=int((parallel or {}).get("max_workers", 4)),
            use_cache=True,
        )
        # Infer out_dir from export stage or report
        out_dir = None
        if B.report.get("viz", {}).get("overlay_paths"):
            import os

            out_path = B.report["viz"]["overlay_paths"][0]
            out_dir = os.path.dirname(out_path)
        return bundle_id, {"report": B.report, "out_dir": out_dir}

    results: Dict[str, Dict[str, Any]] = {}
    par = parallel or {}
    if par.get("enable"):
        max_workers = int(par.get("max_workers", 4))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_process, bid, frs): bid for bid, frs in grouped.items()}
            for fut in as_completed(futs):
                bid, res = fut.result()
                results[bid] = res
    else:
        for bid, frs in grouped.items():
            b, res = _process(bid, frs)
            results[b] = res
    return results
