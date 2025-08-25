"""
Tests for the model loading utility.
"""

import inspect
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import yaml
from torch import nn

from development.utils.loading.model_loader import load_model_from_checkpoint


# 1. Define a dummy model class for use in tests
class DummyModel(nn.Module):
    def __init__(self, input_dim: int = 10, output_dim: int = 5):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# 2. Create a pytest fixture to set up a temporary test environment
@pytest.fixture
def test_env(tmp_path: Path) -> dict:
    """
    Sets up a temporary directory structure, config files, and a dummy checkpoint
    for testing the model loader.
    """
    project_root = tmp_path
    (project_root / "pyproject.toml").touch()

    # Define parameters for the test
    exp_name = "test_experiment"
    version = "version_01"
    model_params = {"input_dim": 20, "output_dim": 8}

    # Create a dummy model and its state_dict with the 'model.' prefix
    model = DummyModel(**model_params)
    original_state_dict = model.state_dict()
    prefixed_state_dict = {"model." + k: v for k, v in original_state_dict.items()}
    checkpoint_data = {"state_dict": prefixed_state_dict}

    # Create the directory structure for a fake caller script and logs
    caller_dir = project_root / "development" / "test_cat" / exp_name
    caller_dir.mkdir(parents=True, exist_ok=True)
    caller_path = caller_dir / "test_caller.py"
    caller_path.touch()

    log_dir = project_root / "tb_logs" / exp_name / version
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create a fake hparams.yaml file
    hparams = {"lit_module": {"model": model_params}}
    with open(log_dir / "hparams.yaml", "w") as f:
        yaml.dump(hparams, f)

    # Create a fake checkpoint file
    checkpoint_path = checkpoint_dir / "test.ckpt"
    torch.save(checkpoint_data, checkpoint_path)

    # Return a dictionary of paths and objects for the tests to use
    return {
        "project_root": project_root,
        "caller_path": str(caller_path),
        "version_name": version,
        "DummyModel": DummyModel,
        "original_state_dict": original_state_dict,
    }


# 3. Helper function to mock dependencies
def mock_dependencies(monkeypatch: pytest.MonkeyPatch, test_env: dict):
    """Mocks dependencies like inspect.stack and find_project_root."""
    # Mock inspect.stack to simulate the call from our dummy script
    mock_frame = MagicMock()
    mock_frame.filename = test_env["caller_path"]
    monkeypatch.setattr(inspect, "stack", lambda: [MagicMock(), mock_frame])

    # Mock find_project_root to always return our temporary project root
    monkeypatch.setattr(
        "development.utils.loading.model_loader.find_project_root",
        lambda: test_env["project_root"],
    )


# 4. Test cases
def test_load_model_successfully(monkeypatch: pytest.MonkeyPatch, test_env: dict):
    """Tests the successful loading of a model from a checkpoint."""
    mock_dependencies(monkeypatch, test_env)

    loaded_model = load_model_from_checkpoint(
        model_cls=test_env["DummyModel"],
        version_name=test_env["version_name"],
    )

    assert isinstance(loaded_model, test_env["DummyModel"])
    assert not loaded_model.training  # Should be in evaluation mode

    # Verify that the loaded weights match the original weights
    original_sd = test_env["original_state_dict"]
    loaded_sd = loaded_model.state_dict()
    assert original_sd.keys() == loaded_sd.keys()
    for key in original_sd:
        assert torch.equal(original_sd[key], loaded_sd[key])


def test_raises_file_not_found(monkeypatch: pytest.MonkeyPatch, test_env: dict):
    """Tests that FileNotFoundError is raised for missing files or directories."""
    mock_dependencies(monkeypatch, test_env)

    # Test with a missing checkpoint file
    ckpt_path = next((test_env["project_root"] / "tb_logs").glob("**/*.ckpt"))
    os.remove(ckpt_path)

    with pytest.raises(FileNotFoundError, match="No checkpoint file"):
        load_model_from_checkpoint(model_cls=test_env["DummyModel"], version_name=test_env["version_name"])

    # Test with a missing hparams.yaml (after restoring a dummy ckpt file)
    ckpt_path.touch()
    hparams_path = next((test_env["project_root"] / "tb_logs").glob("**/hparams.yaml"))
    os.remove(hparams_path)

    with pytest.raises(FileNotFoundError, match="hparams.yaml not found"):
        load_model_from_checkpoint(model_cls=test_env["DummyModel"], version_name=test_env["version_name"])


def test_raises_value_error_for_bad_path(monkeypatch: pytest.MonkeyPatch, test_env: dict):
    """Tests that ValueError is raised for an invalid caller path."""
    # Simulate a call from a path that does not match the expected structure
    mock_frame = MagicMock()
    mock_frame.filename = str(test_env["project_root"] / "another_dir" / "script.py")
    monkeypatch.setattr(inspect, "stack", lambda: [MagicMock(), mock_frame])

    monkeypatch.setattr(
        "development.utils.loading.model_loader.find_project_root",
        lambda: test_env["project_root"],
    )

    with pytest.raises(ValueError, match="Could not infer experiment name"):
        load_model_from_checkpoint(model_cls=test_env["DummyModel"], version_name=test_env["version_name"])


def test_raises_value_error_for_bad_hparams(monkeypatch: pytest.MonkeyPatch, test_env: dict):
    """Tests that ValueError is raised for an hparams file missing model config."""
    mock_dependencies(monkeypatch, test_env)

    # Overwrite hparams.yaml with content that is missing the model parameters
    hparams_path = next((test_env["project_root"] / "tb_logs").glob("**/hparams.yaml"))
    with open(hparams_path, "w") as f:
        yaml.dump({"other_key": "value"}, f)

    with pytest.raises(ValueError, match="Could not find model parameters"):
        load_model_from_checkpoint(model_cls=test_env["DummyModel"], version_name=test_env["version_name"])
