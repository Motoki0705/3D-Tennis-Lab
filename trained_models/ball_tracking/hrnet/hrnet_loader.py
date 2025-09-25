# hrnet_loader.py
"""Utility loader for the HRNet-based tennis ball tracking stack.

This module mirrors the helper pattern used in :mod:`trained_models`. Rather
than invoking the development runner, it composes the copied Hydra configs,
instantiates the TrackNetV2 detector and paired tracker from WASB-SBDT, and
returns ready-to-use components along with the resolved config.

Quickstart
----------
>>> from trained_models.ball_tracking.hrnet.hrnet_loader import (
...     HRNetLoadConfig,
...     load_hrnet_with_ckpt,
... )
>>> cfg = HRNetLoadConfig.from_yaml("hrnet_config.yaml")
>>> detector, tracker, transform, device, resolved_cfg = load_hrnet_with_ckpt(cfg)
>>> print(resolved_cfg.model.frames_in)

The detector expects ``frames_in`` consecutive frames processed with
``transform`` and concatenated along the channel axis, matching the development
runner's inference path.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, open_dict

# Third-party implementation of TrackNetV2 + tracker lives here.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
WASB_SRC = PROJECT_ROOT / "third_party" / "WASB-SBDT" / "src"
if str(WASB_SRC) not in sys.path:
    sys.path.insert(0, str(WASB_SRC))

from dataloaders import build_img_transforms  # type: ignore  # noqa: E402
from detectors import build_detector  # type: ignore  # noqa: E402
from trackers import build_tracker  # type: ignore  # noqa: E402

DEFAULT_CONFIG_DIR = PROJECT_ROOT / "trained_models" / "ball_tracking" / "hrnet" / "configs"

__all__ = [
    "HRNetLoadConfig",
    "compose_hrnet_cfg",
    "load_hrnet_with_ckpt",
]


@dataclass
class HRNetLoadConfig:
    """Configuration container for :func:`load_hrnet_with_ckpt`.

    Parameters
    ----------
    checkpoint_path:
        Absolute or relative path to the TrackNetV2 checkpoint file. If left
        ``None`` the loader falls back to ``detector.model_path`` in the Hydra
        config. At least one of these must reference an existing file.
    config_dir:
        Directory containing the Hydra config tree. The default points to the
        curated copy under ``trained_models`` so the loader is self-contained.
    config_name:
        Name of the root Hydra config to compose (``"detect"`` mirrors the
        development entry point).
    overrides:
        Optional Hydra override strings applied during composition, e.g.
        ``("runner.vis_result=True",)``.
    device:
        Torch device identifier. The TrackNetV2 detector only supports
        ``"cuda"``; other values raise immediately.
    gpu_ids:
        GPU indices forwarded to WASB's ``nn.DataParallel`` wrapper. The first
        entry also determines the device object returned to the caller.
    """

    checkpoint_path: str | None = None
    config_dir: str = str(DEFAULT_CONFIG_DIR)
    config_name: str = "detect"
    overrides: tuple[str, ...] = ()
    device: str = "cuda"
    gpu_ids: tuple[int, ...] = (0,)

    @classmethod
    def from_yaml(cls, path: str) -> HRNetLoadConfig:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)


def load_hrnet_with_ckpt(
    load_cfg: HRNetLoadConfig,
) -> tuple[Any, Any, Callable[[Any], torch.Tensor], torch.device, DictConfig]:
    """Instantiate the HRNet detector, tracker, and preprocessing stack.

    Mirrors the loading pattern of other helpers in ``trained_models``: validate
    the runtime configuration, compose Hydra defaults, build the underlying
    WASB modules, and return fully configured objects suitable for inference.

    Returns
    -------
    detector:
        Instance of ``TracknetV2Detector`` from the WASB-SBDT package.
    tracker:
        Tracker that pairs with the detector (typically ``OnlineTracker``).
    transform:
        Callable preprocessing pipeline (``PIL.Image`` -> normalised
        ``torch.FloatTensor``) identical to the test-time transform used at
        training.
    device:
        CUDA device selected for inference.
    composed_cfg:
        Fully resolved Hydra :class:`~omegaconf.DictConfig`, handy when passing
        configuration values to other utilities.

    Raises
    ------
    FileNotFoundError
        If the config directory or checkpoint file cannot be located.
    RuntimeError
        If CUDA is requested but unavailable.
    ValueError
        If runtime overrides (device, GPU IDs, checkpoint path) are inconsistent.
    """

    cfg = compose_hrnet_cfg(load_cfg.config_dir, load_cfg.config_name, load_cfg.overrides)
    _apply_runtime_overrides(cfg, load_cfg)
    OmegaConf.resolve(cfg)

    device = _select_device(str(cfg.runner.device))

    # build_img_transforms returns (train, test); inference uses the test one.
    _, transform = build_img_transforms(cfg)
    detector = build_detector(cfg)
    tracker = build_tracker(cfg)

    return detector, tracker, transform, device, cfg


def compose_hrnet_cfg(
    config_dir: str,
    config_name: str,
    overrides: Sequence[str] | tuple[str, ...],
) -> DictConfig:
    """Compose the Hydra config tree used for HRNet inference."""

    config_dir_path = Path(config_dir).expanduser().resolve()
    if not config_dir_path.is_dir():
        raise FileNotFoundError(f"Config directory not found: {config_dir_path}")

    hydra = GlobalHydra.instance()
    if hydra.is_initialized():
        hydra.clear()

    with initialize_config_dir(config_dir=str(config_dir_path), job_name="hrnet_loader"):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    return cfg


def _apply_runtime_overrides(cfg: DictConfig, load_cfg: HRNetLoadConfig) -> None:
    """Mutate the composed config with runtime-specific overrides and checks."""

    with open_dict(cfg):
        if load_cfg.device != "cuda":
            raise ValueError("HRNet TrackNetV2 detector supports only device='cuda'.")
        cfg.runner.device = load_cfg.device

        gpu_ids = [int(idx) for idx in load_cfg.gpu_ids]
        if not gpu_ids:
            raise ValueError("gpu_ids must contain at least one GPU index.")
        cfg.runner.gpus = gpu_ids

        checkpoint_candidate = load_cfg.checkpoint_path or cfg.detector.get("model_path")
        if not checkpoint_candidate:
            raise ValueError("Provide a checkpoint via HRNetLoadConfig.checkpoint_path or the config YAML.")
        checkpoint_path = Path(checkpoint_candidate).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        cfg.detector.model_path = str(checkpoint_path)


def _select_device(preferred: str) -> torch.device:
    """Return the torch device while enforcing TrackNetV2 constraints."""

    if preferred != "cuda":
        raise ValueError("HRNet TrackNetV2 detector can only run on CUDA GPUs.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return torch.device("cuda")
