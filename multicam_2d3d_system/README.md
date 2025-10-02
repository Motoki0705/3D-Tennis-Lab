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
3. **detection**: run HRNet ball, RT-DETR player, ViT-Pose keypoints, and DINO-FPN court detectors.
4. **tracking**: associate per-camera detections into lightweight tracks.
5. **triangulation**: reconstruct smoothed 3D trajectories and align to the court frame.
6. **export/viz**: persist results (Parquet/JSON) and provide visualization hooks.

Refer to `configs/` for Hydra entry points and `scripts/` for runnable presets.

## CLI Usage

The entrypoint prefers [Hydra](https://hydra.cc/) and `omegaconf` for configuration. Install them via:

```bash
pip install hydra-core omegaconf
```

Execute individual stages with the helper scripts (overrides are forwarded to Hydra):

```bash
bash scripts/run_detect_2d.sh workspace.root=/data/matches
bash scripts/run_track_2d.sh detection.tracker.method=simple_centroid
bash scripts/run_triangulate_3d.sh triangulation.smoothing.filter=kalman
```

If Hydra is unavailable, the runner falls back to a minimal parser that accepts a `--config` path and `key=value` overrides, e.g.:

```bash
python3 -m multicam_2d3d_system.src.main --config multicam_2d3d_system/configs/config.yaml pipeline=full
```
