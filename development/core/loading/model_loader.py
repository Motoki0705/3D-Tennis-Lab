"""
This module provides a utility function to load trained models from checkpoints.
"""

import inspect
import logging
from pathlib import Path
from typing import TypeVar

import torch
import yaml
from torch import nn

# Define a generic type for nn.Module subclasses
T = TypeVar("T", bound=nn.Module)

# Set up a logger for informative output
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def find_project_root(marker: str = "pyproject.toml") -> Path:
    """
    Finds the project root directory by searching upwards for a marker file.

    Args:
        marker: The filename to look for to identify the root (e.g., 'pyproject.toml').

    Returns:
        The path to the project root directory.

    Raises:
        FileNotFoundError: If the project root cannot be located.
    """
    path = Path.cwd().resolve()
    while path != path.parent:
        if (path / marker).exists():
            return path
        path = path.parent
    raise FileNotFoundError(f"Project root with marker '{marker}' not found.")


def load_model_from_checkpoint(model_cls: type[T], version_name: str) -> T:
    """
    Loads a model from a specified training checkpoint.

    This function infers the experiment name from the file path of the caller.
    It then locates the corresponding TensorBoard log directory to load hyperparameters
    and model weights from the specified checkpoint version.

    The state dictionary keys are adjusted by removing the 'model.' prefix
    before loading them into the model instance.

    Args:
        model_cls: The model class to be instantiated (e.g., ViTHeatmap).
        version_name: The specific version of the checkpoint to load,
                      corresponding to the directory name in tb_logs
                      (e.g., "version_12").

    Returns:
        An instance of the model class with loaded weights, set to evaluation mode.

    Raises:
        ValueError: If the experiment name cannot be inferred from the path,
                    or if model parameters are not found in hparams.yaml.
        FileNotFoundError: If the required log directory, hparams.yaml, or
                           checkpoint file cannot be found.
    """
    try:
        # Infer experiment name from the caller's file path
        caller_frame = inspect.stack()[1]
        caller_path = Path(caller_frame.filename).resolve()
        parts = caller_path.parts
        dev_index = parts.index("development")
        if len(parts) <= dev_index + 2:
            raise ValueError(
                "The script calling this function must be located in a directory "
                "like 'development/<category>/<experiment_name>/'."
            )
        experiment_name = parts[dev_index + 2]
        logger.info(f"Inferred experiment name: {experiment_name}")
    except (ValueError, IndexError) as e:
        raise ValueError(
            "Could not infer experiment name from the caller's path. "
            f"Ensure the file structure is correct. Path: {caller_path}"
        ) from e

    # Construct paths to log and checkpoint files
    project_root = find_project_root()
    log_dir = project_root / "tb_logs" / experiment_name / version_name
    hparams_path = log_dir / "hparams.yaml"
    checkpoint_dir = log_dir / "checkpoints"

    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
    if not hparams_path.exists():
        raise FileNotFoundError(f"hparams.yaml not found in {log_dir}")
    if not checkpoint_dir.exists() or not any(checkpoint_dir.glob("*.ckpt")):
        raise FileNotFoundError(f"No checkpoint file (.ckpt) found in {checkpoint_dir}")

    # Load hyperparameters from the yaml file
    with open(hparams_path) as f:
        hparams = yaml.safe_load(f)

    # Extract model parameters, often nested under 'model' or 'lit_module.model'
    model_params = hparams.get("model", hparams.get("lit_module", {}).get("model", {}))
    if not model_params:
        raise ValueError(
            "Could not find model parameters in hparams.yaml. "
            "Expected them under the 'model' or 'lit_module.model' key."
        )

    # Instantiate the model with the loaded hyperparameters
    model = model_cls(**model_params)

    # Load the checkpoint file
    checkpoint_path = next(checkpoint_dir.glob("*.ckpt"))
    logger.info(f"Loading model weights from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = checkpoint["state_dict"]

    # Create a new state dictionary, removing the 'model.' prefix from keys
    new_state_dict = {}
    prefix = "model."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_state_dict[key[len(prefix) :]] = value

    # Load the cleaned state dictionary into the model
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        logger.warning(f"Missing keys in state_dict: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")

    # Set the model to evaluation mode
    model.eval()

    logger.info(f"Successfully loaded model '{model_cls.__name__}' from {checkpoint_path}.")
    return model
