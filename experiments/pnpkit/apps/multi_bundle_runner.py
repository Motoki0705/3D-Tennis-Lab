from __future__ import annotations

import os
import hydra
from omegaconf import DictConfig, OmegaConf

from experiments.pnpkit.src.pipeline.base import Bundle
from experiments.pnpkit.src.pipeline.bundler import group_frames
from experiments.pnpkit.src.utils.hydra_tools import instantiate_stages
from experiments.pnpkit.src.pipeline.executor import run_dag
from experiments.pnpkit.src.io.reader import adapt_frames, load_camera_index
from experiments.pnpkit.src.io.court import court3d_from_cfg


def _load_frames(data_cfg):
    frames, warnings = adapt_frames(data_cfg)
    return frames, warnings


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Load frames
    frames, global_warnings = _load_frames(cfg.data)
    # Group into bundles
    bundling = cfg.get("bundling") or {}
    strategy = bundling.get("strategy", "by_prefix")
    params = OmegaConf.to_container(bundling.get("params", {}), resolve=True) or {}
    grouped = group_frames(frames, strategy=strategy, **params)

    # Prepare court3d and K defaults
    from omegaconf import OmegaConf as OC

    court3d = court3d_from_cfg(OC.to_container(cfg.courts, resolve=True))

    # camera index mapping if provided
    K_default = cfg.cameras
    mapping = None
    if cfg.data.get("camera_index"):
        mapping = load_camera_index(cfg.data["camera_index"])

    # Cameras granularity: by_bundle | by_scene | global
    gran = str(cfg.get("cameras_granularity", "by_bundle"))

    # Build stage list (skip LoadInputs if present)
    stage_cfgs = []
    for sc in cfg.pipeline.stages:
        tgt = sc.get("_target_", "")
        if tgt.endswith("s00_load_inputs.LoadInputs"):
            continue
        stage_cfgs.append(sc)
    stages = instantiate_stages(stage_cfgs)

    # Run per bundle
    # Optionally resplit bundles if multiple K mappings appear inside a bundle
    def choose_intr_path(img_path: str) -> str | None:
        if not mapping:
            return None
        bn = os.path.basename(img_path)
        for pref, intr_path in mapping.items():
            if bn.startswith(pref):
                return intr_path
        return None

    # Auto-split per K if mixed inside a bundle
    grouped2 = {}
    for bundle_id, frs in grouped.items():
        if gran == "global" or not mapping:
            grouped2[bundle_id] = frs
            continue
        buckets = {}
        for fr in frs:
            intrp = choose_intr_path(fr.image_path) or "__default__"
            buckets.setdefault(intrp, []).append(fr)
        if len(buckets) == 1:
            grouped2[bundle_id] = frs
        else:
            # split into sub-bundles
            for k_id, fr_list in buckets.items():
                suffix = k_id.split("/")[-1].split("\\")[-1] if k_id and k_id != "__default__" else "default"
                grouped2[f"{bundle_id}{suffix}"] = fr_list

    for bundle_id, frs in grouped2.items():
        # Determine K
        K_cfg = K_default
        if gran == "global":
            K_cfg = K_default
        elif mapping and frs:
            # choose by first frame in this bundle (post-split ensures consistency)
            chosen = choose_intr_path(frs[0].image_path)
            if chosen:
                from omegaconf import OmegaConf as OC

                K_cfg = OC.load(chosen)

        B = Bundle(bundle_id=str(bundle_id), court3d=court3d, frames=frs)
        # Attach K in the same dict schema used elsewhere
        from experiments.pnpkit.src.core.camera import Intrinsics

        Kintr = Intrinsics.from_dict(OC.to_container(K_cfg, resolve=True))
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

        # Attach any global warnings collected from I/O
        if global_warnings:
            B.report.setdefault("warnings", {})
            B.report["warnings"].setdefault("io", []).extend(global_warnings)

        # Execute via DAG (enables skip/branch/cache)
        B = run_dag(B, stages, parallel=False, use_cache=True)
        print(f"bundle={bundle_id} -> report keys: {list(B.report.keys())}")


if __name__ == "__main__":
    main()
