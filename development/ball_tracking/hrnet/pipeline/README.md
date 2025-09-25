# HRNet Ball Tracking Pipeline

This directory orchestrates the semi-automatic tennis-ball annotation workflow that powers the 3D Tennis Lab dataset. The CLI entry point for all actions is `python -m development.ball_tracking.hrnet.pipeline`.

## Workflow

1. **Scan & index videos** (`scanner.py`, `state.py`)

   - `pipeline run` walks `paths.videos_root` and registers every supported video (`.mp4`, `.mov`, `.mkv`, `.avi`) in the SQLite state database located at `paths.state_db`.
   - Registered rows track `status` (`queued` -> `running` -> `done`/`failed`), `last_processed_frame`, and `total_frames` so interrupted jobs can resume.

2. **Frame-wise inference** (`inference.py`, `parquet_io.py`)

   - Each pending video is decoded with OpenCV. Frames are warped to the HRNet input size (`model.inp_width x model.inp_height`) and normalized before being stacked into tensors of shape `(1, frames_in * 3, inp_height, inp_width)` where `frames_in` defaults to 3.
   - The WASB-SBDT detector + tracker pair (Hydra config under `development/ball_tracking/hrnet/configs`) produces per-frame ball center predictions. Tracker output is projected back to the source frame, yielding per-frame records: `frame_idx`, timestamp `t` (seconds), `has_ball`, `xc`, `yc` (pixel coordinates in the original frame), and `conf` (confidence).
   - Records are periodically appended to `<paths.inference_dir>/<video_id>.parquet`. If inference fails, the status is set to `failed` leaving the existing parquet untouched.

3. **Clip extraction** (`clip_extractor/`)

   - The parquet DataFrame is filtered to rows where `has_ball == True` and grouped into clips according to the configured strategy:
     - `contiguous` (default) joins detections separated by gaps up to `max_gap_frames`; clips shorter than `min_clip_len` are discarded.
     - `clustering` groups detections by spatio-temporal density (DBSCAN) using the selected feature columns.
   - Each `Clip` contains ordered `ClipFrame` objects preserving `frame_idx`, `t`, `xc`, `yc`, and `conf` for downstream export.

4. **Clip export packaging** (`annotation.py`)

   - New clips are compared against existing metadata to avoid duplicates, then exported beneath `paths.images_dir/<game_id>/Clip*/` as JPEG frames plus a COCO-style JSON annotation in `paths.ann_clips_dir`.
   - Clip JSON stores the original video metadata (`video_id`, `start_frame`, `end_frame`) and defaults `accepted` to `false` for later review.

5. **Manual review & merge**
   - Reviewers call `pipeline accept <game_id/ClipN>` or `pipeline reject ...` to toggle the `accepted` flag in-place.
   - `pipeline finalize` gathers every accepted clip JSON and merges them into `paths.ann_final`, re-indexing image and annotation IDs to produce the training-ready dataset artifact.

## Configuration

Hydra loads `conf/config.yaml`. Override any value via CLI suffixes (e.g. `python -m ... run logging.level=DEBUG inference.save_every_n_frames=50`). The top-level sections are:

- `paths`: Input video root, export folders, parquet output, final annotation target, and state database location. Paths may reference each other using `${paths.*}` interpolation.
- `inference`: High-level pipeline settings. Key fields:
  - `model`/`method`: Informational tags recorded alongside hydra overrides.
  - `batch_size`: Passed to the detector runner.
  - `save_every_n_frames`: Flush frequency for parquet writes.
  - `model_path`: Checkpoint path propagated to the detector config.
- `clip_extractor`: Selects the extraction strategy and its parameters (see step 3). Switching to clustering requires scikit-learn at runtime.
- `export`: Controls whether raw frames are written and the JPEG quality used for exported clips.
- `merge`: Currently only `only_accept`, guarding against accidental inclusion of un-reviewed clips.
- `logging`: Global logging level applied by `cli.py`.

### Detector stack configuration

Hydra composes the detector runtime directly from the config groups declared in `defaults`:

- `runner`: Execution device, visualization toggles, and evaluation thresholds.
- `model`: WASB/HRNet architecture parameters (frames in/out, input/output resolution, scales).
- `detector`: TrackNetV2-specific knobs, including interpolation to `inference.model_path` for checkpoints.
- `transform`: Augmentation toggles for train/test pipelines (kept deterministic for inference).
- `tracker`: Online tracker thresholds and displacement settings.

Customize any of these via CLI overrides (e.g. `python -m ... run runner.device=cpu detector.postprocessor.score_threshold=0.3`).

## Data shape flow

| Stage           | Representation                 | Structure                                                 | Notes                                                                                         |
| --------------- | ------------------------------ | --------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Video decode    | `numpy.ndarray`                | `(frame_height, frame_width, 3)` uint8 (BGR)              | Raw frame from OpenCV.                                                                        |
| Model input     | `torch.Tensor`                 | `(1, frames_in*3, inp_height, inp_width)` float32         | Concatenation of `frames_in` normalized frames.                                               |
| Detector output | Python dict                    | Keys: `x`, `y`, `score`, `visi`                           | Tracker-projected ball center in source pixels.                                               |
| Inference log   | `pandas.DataFrame` -> parquet  | Columns: `frame_idx`, `t`, `has_ball`, `xc`, `yc`, `conf` | One row per processed frame.                                                                  |
| Clip object     | `Clip(frames=[ClipFrame ...])` | Ordered frames with the same columns as parquet           | Gaps > `max_gap_frames` or short clips removed.                                               |
| Export JSON     | COCO-like dict                 | `images`, `annotations`, `metadata`, `accepted`           | `annotations[*].keypoints = [x, y, visibility]` with visibility `2` when the ball is present. |

## Command quick reference

- `python -m development.ball_tracking.hrnet.pipeline run` – process all pending videos end-to-end.
- `python -m development.ball_tracking.hrnet.pipeline accept <game_id/ClipN>` – mark a clip for inclusion in the final merge.
- `python -m development.ball_tracking.hrnet.pipeline reject <game_id/ClipN>` – exclude a clip.
- `python -m development.ball_tracking.hrnet.pipeline finalize` – build the consolidated annotation file at `paths.ann_final`.

State and export directories are safe to remove or relocate manually, but remember to update the corresponding `paths.*` entries (or override them via CLI) before running the pipeline again.
