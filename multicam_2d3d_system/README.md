# Multicam 2D→3D System

This package orchestrates the end-to-end workflow that converts multi-camera 2D detections into synchronized 3D trajectories. The module consumes trained 2D detectors from `development/core` and exports normalized 3D data that can seed downstream 3D tennis models.

## Goals

- ingest per-camera detections and tracking output
- recover per-camera calibration and global synchronization
- perform geometry-aware triangulation with temporal smoothing
- export analyzable datasets and diagnostics for 3D training

## High-Level Pipeline

1. **sync**: harmonize timestamps across cameras.
2. **calibration**: solve for intrinsics/extrinsics and refine with bundle adjustment.
3. **detection/track**: run batched 2D inference and multi-object tracking per camera.
4. **triangulation**: reconstruct 3D positions with temporal smoothing and court alignment.
5. **export/viz**: persist results (Parquet/JSON) and provide visualization hooks.

Refer to `configs/` for Hydra entry points and `scripts/` for runnable presets.

## CLI Usage

The entrypoint prefers [Hydra](https://hydra.cc/) and `omegaconf` for configuration. Install them via:

```bash
pip install hydra-core omegaconf
```

Execute individual stages with the helper scripts (overrides are forwarded to Hydra):

```bash
bash scripts/run_detect_2d.sh workspace.root=/data/matches
bash scripts/run_track_2d.sh detection.tracker.method=bytetrack
bash scripts/run_triangulate_3d.sh triangulation.method=robust
```

If Hydra is unavailable, the runner falls back to a minimal parser that accepts a `--config` path and `key=value` overrides, e.g.:

```bash
python3 -m multicam_2d3d_system.src.main --config multicam_2d3d_system/configs/config.yaml pipeline=full
```
