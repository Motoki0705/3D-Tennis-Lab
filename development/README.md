# Development Directory: The Experiment Lab

## 1. Philosophy and Purpose

This directory is the **primary workspace for research and development**. Think of it as a laboratory where we experiment with and develop the individual model components that make up the 3D Tennis Analysis System.

The core philosophy is to enable **rapid iteration and trial-and-error** while maintaining a structured and reproducible workflow. The key to achieving this balance lies in the strategic use of the `utils` directory.

## 2. Directory Structure

### 2.1. Experiment-centric Organization

The `development` directory is organized by the specific component being developed (e.g., `court_pose`, `ball_tracking`). Each component directory contains further subdirectories for each specific experiment or approach.

```
development/
├── court_pose/
│   ├── 01_vit_heatmap/  # A specific experiment for court pose estimation
│   └── 02_another_approach/
├── ball_tracking/
│   └── 01_detect_then_track/
└── utils/
    ├── callbacks/
    ├── lightning/
    ├── loss/
    └── transformers/
```

Each experiment directory, like `01_vit_heatmap`, is a self-contained unit containing its own models, configurations, and training scripts. This allows for isolated development without affecting other experiments.

### 2.2. Inside an Experiment Directory (`01_vit_heatmap` example)

A typical experiment directory has the following structure:

```
01_vit_heatmap/
├── configs/
│   └── vit_heatmap_v1.yaml  # Hydra configuration for this experiment
├── model/
│   ├── __init__.py
│   ├── decoder/
│   ├── vit_encoder.py
│   └── vit_heatmap.py       # The main model definition
├── tests/
│   └── ...                  # Unit tests for this specific experiment
├── datamodule.py            # Pytorch Lightning DataModule
├── lit_module.py            # Pytorch Lightning LightningModule
└── train.py                 # The main training script
```

## 3. The Role of `utils`: Our Shared Toolkit

As the number of experiments grows, it's crucial to avoid code duplication and maintain consistency. The `development/utils` directory serves as a **shared toolkit of reusable and well-tested components** for all experiments.

**Why is `utils` so important?**

-   **Reduces Boilerplate**: It abstracts away common, repetitive code (e.g., training loops, data splitting, optimizer setup).
-   **Enforces Consistency**: It ensures that different experiments use the same core logic for tasks like loss calculation or data augmentation.
-   **Accelerates Development**: Instead of rewriting everything, you can import and use pre-built, reliable components, allowing you to focus on the unique aspects of your experiment.

### 3.1. Anatomy of `utils`

The `utils` directory is categorized by functionality:

-   `utils/lightning/`: Contains base classes for Pytorch Lightning, like `BaseLitModule` and `BaseDataModule`.
-   `utils/loss/`: Provides a registry and implementations for various loss functions (`FocalLoss`, `KLDivLoss`, etc.).
-   `utils/callbacks/`: Holds custom Pytorch Lightning callbacks, such as `HeatmapImageLogger` for visualizing results in TensorBoard.
-   `utils/transformers/`: Contains common data augmentation and transformation pipelines using `albumentations`.

### 3.2. How `utils` are used in an Experiment (`01_vit_heatmap`)

Let's see how our `01_vit_heatmap` experiment leverages the shared `utils` toolkit.

#### **`utils/lightning` → `lit_module.py` & `datamodule.py`**

-   **What it does**: `BaseLitModule` handles the entire generic training/validation/test loop and the basic `configure_optimizers` logic. `BaseDataModule` handles the splitting of datasets.
-   **How it's used**:
    -   `CourtLitModule` in `lit_module.py` **inherits** from `BaseLitModule`. This means we only need to implement the logic specific to our experiment (like freezing the ViT and setting up parameter-group-specific learning rates), not the entire training loop.
    -   `CourtDataModule` in `datamodule.py` **inherits** from `BaseDataModule`, offloading the responsibility of splitting the dataset into train/validation/test sets.

    ```python
    # In development/court_pose/vit_heatmap_01/lit_module.py
    from ...utils.lightning.base_lit_module import BaseLitModule

    class CourtLitModule(BaseLitModule):
        def __init__(self, config):
            # ... setup model, loss, metrics ...
            super().__init__(config=config, model=model, loss_fn=loss_fn, metric_fns=metric_fns)
            # ... custom logic ...
    ```

#### **`utils/loss` → `lit_module.py` & `configs/vit_heatmap_v1.yaml`**

-   **What it does**: The `loss_registry` allows us to dynamically select and instantiate any registered loss function simply by specifying its name in the config file.
-   **How it's used**:
    -   `lit_module.py` calls `loss_registry.get()` to fetch the loss function specified in the YAML file. This makes the code clean and removes the need for `if/elif` statements for each loss type.
    -   This allows us to run experiments with different loss functions (`mse`, `focal`, etc.) just by changing one line in `vit_heatmap_v1.yaml`.

    ```python
    # In development/court_pose/vit_heatmap_01/lit_module.py
    from ...utils.loss import loss_registry

    def _get_loss_fn(self, loss_config):
        return loss_registry.get(loss_config.name, **loss_config.params)

    # In development/court_pose/vit_heatmap_01/configs/vit_heatmap_v1.yaml
    training:
      loss:
        name: "focal" # Simply change this string to switch loss functions
        params: {alpha: 0.5}
    ```

#### **`utils/callbacks` & `utils/transformers` → `train.py`**

-   **What it does**: `HeatmapImageLogger` provides a reusable way to log image predictions to TensorBoard. `keypoint_transformer` provides standardized data augmentation pipelines.
-   **How it's used**:
    -   In `train.py`, we simply instantiate `HeatmapImageLogger` and `prepare_transforms` and pass them to the Pytorch Lightning `Trainer` and `DataModule`, respectively. This keeps the main training script clean and focused on orchestrating the training process.

    ```python
    # In development/court_pose/vit_heatmap_01/train.py
    from ...utils.callbacks.heatmap_logger import HeatmapImageLogger
    from ...utils.transformers.keypoint_transformer import prepare_transforms

    # ...
    heatmap_logger = HeatmapImageLogger(...)
    train_transform, val_transform = prepare_transforms(...)
    datamodule = CourtDataModule(config=config, train_transforms=train_transform, ...)
    trainer = pl.Trainer(callbacks=[heatmap_logger, ...])
    # ...
    ```

## 4. Development Workflow

Follow these steps to ensure a smooth and efficient development process.

1.  **Check `utils` First**: Before starting a new experiment, always check the `development/utils` directory to see if components you need already exist.

2.  **Create an Experiment Directory**: The best way to start a new experiment is to copy an existing one (like `01_vit_heatmap`) as a template. This gives you a solid starting point with all the necessary files.

3.  **Develop & Experiment**: This is where you innovate. Modify the model architecture in `model/`, tweak parameters in `configs/*.yaml`, and adjust the logic in `lit_module.py`. Because you are leveraging `utils`, you can focus on these high-impact changes.

4.  **Contribute Back to `utils`**: If you develop a new, reusable component during your experiment (e.g., a new data augmentation technique, a new generic callback, or a new loss function), **refactor it and move it into the `utils` directory**. This is crucial for enriching our shared toolkit and benefiting future experiments.

5.  **Graduate the Best Model**: Once you have identified a successful model, its artifacts must be "graduated" to the central `trained_models` directory.
    -   Copy the final model weights (e.g., `best_model.pth`).
    -   Copy the exact configuration file used for training (`config.yaml`).
    -   (Optional) Copy any relevant training logs or evaluation results.

This process ensures that the `development/` directory remains a pure experimental space, while the `trained_models/` directory serves as a stable, centralized repository for finalized models that are ready for integration into the main system.