# Development Directory

This directory is the primary workspace for the development and experimentation of individual model components that make up the 3D Tennis Analysis System.

## Directory Structure

This directory is organized by the specific component being developed (e.g., `court_pose`, `ball_tracking`). Each component directory should contain further subdirectories for each specific experiment or approach.

```
development/
├── court_pose/
│   ├── 01_baseline_cnn/
│   └── 02_advanced_model/
├── ball_tracking/
│   └── 01_detect_then_track/
└── ...
```

## Development Workflow

1.  **Create an Experiment Directory**: Inside the relevant component directory (e.g., `player_analysis`), create a new numbered directory for your experiment (e.g., `03_new_architecture`).
2.  **Develop & Experiment**: Write your model definitions, training scripts, and configuration files within this new directory. This space is self-contained for your trial-and-error process.
3.  **Graduate the Best Model**: Once you have identified a successful model from your experiments, its artifacts must be "graduated" to the central `trained_models` directory.
    *   Copy the final model weights (e.g., `best_model.pth`).
    *   Copy the exact configuration file used for training (`config.yaml`).
    *   (Optional) Copy any relevant training logs or evaluation results.

This process ensures that the `development/` directory remains a pure experimental space, while the `trained_models/` directory serves as a stable, centralized repository for finalized models that are ready for integration into the main system.

## Interaction with Other Directories

*   **Input**: Scripts in this directory may import reusable code (e.g., dataset loaders, utility functions) from the `3d_tennis_system/` directory.
*   **Output**: The final, validated artifacts of your work here are stored in the `/trained_models/` directory.
