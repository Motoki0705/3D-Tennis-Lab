"""Hydra entry point for the multicam 2D→3D reconstruction stack."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from importlib import import_module
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

try:  # pragma: no cover - optional dependency
    from hydra import main as hydra_main
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    hydra_main = None

_LOGGER = logging.getLogger("multicam_2d3d_system")

PIPELINE_TARGETS: dict[str, str] = {
    "detect2d": "multicam_2d3d_system.src.pipelines.detect2d:run",
    "track2d": "multicam_2d3d_system.src.pipelines.track2d:run",
    "triangulate3d": "multicam_2d3d_system.src.pipelines.twoD_to_threeD:run",
    "full": "multicam_2d3d_system.src.pipelines.twoD_to_threeD:run",
}

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(levelname)s] %(name)s: %(message)s",
    )


def _resolve_pipeline(name: str) -> Callable[[DictConfig], None]:
    try:
        target = PIPELINE_TARGETS[name]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown pipeline '{name}'. Known: {sorted(PIPELINE_TARGETS)}") from exc

    module_path, func_name = target.split(":", maxsplit=1)
    module = import_module(module_path)
    try:
        fn = getattr(module, func_name)
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Pipeline target '{target}' is invalid") from exc
    return fn


def _execute(cfg: DictConfig) -> None:
    _configure_logging()
    pipeline_name: str = cfg.get("pipeline", "full")
    _LOGGER.info("Selected pipeline='%s'", pipeline_name)
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    _LOGGER.debug("Resolved config: %s", resolved_cfg)
    pipeline = _resolve_pipeline(pipeline_name)
    pipeline(cfg)


if hydra_main is not None:  # pragma: no branch - simple gating

    @hydra_main(config_path="../configs", config_name="config", version_base=None)
    def _hydra_entry(cfg: DictConfig) -> None:
        _execute(cfg)
else:
    _hydra_entry = None  # type: ignore


def _fallback_entry(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Multicam 2D→3D pipeline runner (simple mode)")
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG_PATH),
        help="Path to the root YAML config",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional key=value overrides (Hydra-style)",
    )
    args = parser.parse_args(argv)

    base_cfg = OmegaConf.load(args.config)
    overrides = OmegaConf.from_cli(args.overrides or [])
    cfg = OmegaConf.merge(base_cfg, overrides)
    _execute(cfg)  # type: ignore[arg-type]


def main(argv: list[str] | None = None) -> None:
    """Delegate to Hydra when available, otherwise use the fallback parser."""
    if _hydra_entry is not None:  # pragma: no branch - simple gating
        _hydra_entry()  # type: ignore[misc]
    else:
        _configure_logging()
        _LOGGER.warning("Hydra is not installed; using fallback argument parser")
        _fallback_entry(argv or sys.argv[1:])


if __name__ == "__main__":
    main()
