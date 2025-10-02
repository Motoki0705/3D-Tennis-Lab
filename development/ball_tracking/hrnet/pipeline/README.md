# HRNet Ball Tracking Annotation Pipeline

## Overview

This pipeline is designed to automate the process of detecting tennis balls in videos, extracting relevant clips, and preparing them for annotation. It uses an HRNet-based model for ball detection, processes videos to identify ball-related events, and provides tools to manage the annotation workflow.

The main goals of this pipeline are:

- To efficiently process large amounts of video data.
- To extract meaningful clips where a ball is present.
- To provide a simple command-line interface for managing the annotation lifecycle (e.g., accepting/rejecting clips).
- To generate a final consolidated annotation file.

## Features

- **Hydra-based Configuration**: Flexible configuration management using Hydra.
- **State Management**: Keeps track of processed videos and their progress using an SQLite database.
- **Extensible Clip Extraction**: Supports different strategies for extracting clips from detection data (e.g., `contiguous`, `clustering`).
- **Annotation Workflow**: Simple commands to `accept`, `reject`, and `finalize` annotations.
- **Parquet-based I/O**: Efficiently stores and reads detection results using Apache Parquet.

## Directory Structure

```
pipeline/
├── conf/                     # Hydra configuration files
│   ├── config.yaml           # Main configuration file
│   ├── dataloader/
│   ├── detector/
│   ├── model/
│   ├── runner/
│   ├── tracker/
│   └── transform/
├── clip_extractor/           # Logic for extracting clips from detection data
├── __main__.py               # Main entry point for the pipeline
├── cli.py                    # Command-line interface definition (using Hydra)
├── pipeline.py               # Core pipeline logic (AnnotationPipeline class)
├── annotation.py             # Annotation handling and exporting functions
├── inference.py              # Inference engine for running the detection model
├── parquet_io.py             # Helper for reading/writing Parquet files
├── scanner.py                # Scans for new videos to process
└── state.py                  # State management using an SQLite repository
```

## Configuration

The pipeline is configured through `conf/config.yaml`. Key configuration sections include:

- **`paths`**: Defines all input and output paths for videos, data, annotations, and the state database.
- **`inference`**: Configures the detection model, including batch size and model path.
- **`clip_extractor`**: Defines the strategy and parameters for extracting clips from detection data.
- **`export`**: Parameters for exporting clips and frames (e.g., image quality).
- **`command`**: The command to be executed by the pipeline (e.g., `run`, `accept`).

Hydra is used for configuration, allowing for easy overrides from the command line.

## Usage

The pipeline is controlled via the command line. The entry point is `__main__.py`.

### Running the Full Pipeline

To run the entire processing pipeline (scan for new videos, run inference, and extract clips):

```bash
python -m development.ball_tracking.hrnet.pipeline
```

or

```bash
python -m development.ball_tracking.hrnet.pipeline command=run
```

This will:

1. Scan the `paths.videos_root` directory for new videos.
2. Run inference on new or partially processed videos.
3. Extract clips based on the `clip_extractor` strategy.
4. Export the clips (frames and metadata) to the `paths.ann_clips_dir`.

### Managing Annotations

You can manage the generated clips using the `accept`, `reject`, and `finalize` commands.

**To accept a clip:**

```bash
python -m development.ball_tracking.hrnet.pipeline command=accept clip=<clip_reference>
```

- `<clip_reference>` is the identifier of the clip (e.g., `game_X/Clip_Y`).

**To reject a clip:**

```bash
python -m development.ball_tracking.hrnet.pipeline command=reject clip=<clip_reference>
```

**To finalize annotations:**

This command merges all "accepted" clips into a single annotation file.

```bash
python -m development.ball_tracking.hrnet.pipeline command=finalize
```

The final annotations will be saved to the path specified by `paths.ann_final`.

## Pipeline Steps (`run` command)

1.  **Scan for Videos**: The `scanner` module checks the `paths.videos_root` directory and compares it against the state database (`state.db`) to find new or unprocessed videos.
2.  **Run Inference**: For each new video, the `InferenceEngine` runs the ball detection model. Detections are saved to a Parquet file in the `paths.inference_dir`. The pipeline can resume from the last processed frame if interrupted.
3.  **Extract Clips**: The `ClipExtractor` processes the Parquet file to identify and extract clips where ball events occur. The extraction logic is defined by the `strategy` in the configuration (e.g., `contiguous`).
4.  **Export Clips**: The extracted clips are then exported. This involves:
    - Saving individual frames as images (if `export.write_frames` is true).
    - Creating a JSON file for each clip with metadata (e.g., frame numbers, video ID).
    - These are saved in subdirectories under `paths.ann_clips_dir`.
