from __future__ import annotations

import logging
import hydra
from omegaconf import DictConfig

from .pipeline import AnnotationPipeline  # <-- keep this


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s][%(levelname)s] %(name)s - %(message)s",
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> int:
    setup_logging(cfg.logging.level)
    pipeline = AnnotationPipeline(cfg)

    if cfg.command == "run":
        pipeline.run()
    elif cfg.command == "accept":
        pipeline.accept(cfg.clip)
    elif cfg.command == "reject":
        pipeline.reject(cfg.clip)
    elif cfg.command == "finalize":
        pipeline.finalize()
    else:
        raise ValueError(f"Unsupported command: {cfg.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
