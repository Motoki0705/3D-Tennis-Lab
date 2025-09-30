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
3. **detection/track**: run batched 2D inference and multi-object tracking per camera. Player
   detections can run either the RT-DETR + ViT-Pose cascade or the single-stage
   DINO-DETR pose model via a configuration flag (see below).
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
bash scripts/run_detect_2d.sh detection.player.pose_mode=single_stage \
    detection.player.single_stage.pose_config=/path/to/dino_detr_pose_config.yaml
bash scripts/run_track_2d.sh detection.tracker.method=bytetrack
bash scripts/run_triangulate_3d.sh triangulation.method=robust
```

Player detection defaults to the two-stage RT-DETR + ViT-Pose pipeline. Switch to the
single-stage DINO-DETR pose detector by setting `detection.player.pose_mode=single_stage`. Override
`detection.player.single_stage.pose_config` (or the two-stage config paths) to point at your
Lightning checkpoints.

If Hydra is unavailable, the runner falls back to a minimal parser that accepts a `--config` path and `key=value` overrides, e.g.:

```bash
python3 -m multicam_2d3d_system.src.main --config multicam_2d3d_system/configs/config.yaml pipeline=full
```
