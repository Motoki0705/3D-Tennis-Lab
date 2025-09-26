"""Compatibility wrapper that redirects to the shared core runner."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_override(args: list[str], key: str, value: str) -> None:
    prefix = f"{key}=" if not key.startswith("+") else key
    if any(arg.startswith(prefix) for arg in args):
        return
    args.append(f"{key}{value}" if key.startswith("+") else f"{key}={value}")


def main() -> None:
    exp_dir = Path(__file__).resolve().parent / "configs"

    overrides = list(sys.argv[1:])
    _ensure_override(overrides, "+experiment_config_dir=", exp_dir.as_posix())
    _ensure_override(overrides, "project", "player_analysis")
    _ensure_override(overrides, "experiment", "dino_detr")

    sys.argv = [sys.argv[0], *overrides]

    from development.core import run as core_run

    core_run.main()


if __name__ == "__main__":
    main()
