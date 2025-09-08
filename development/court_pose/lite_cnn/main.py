from __future__ import annotations

import hydra
from omegaconf import DictConfig

from .runner.train import TrainRunner


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    task = str(getattr(cfg, "task", "train")).lower()
    if task == "train":
        TrainRunner(cfg).run()
    else:
        raise SystemExit(f"Unsupported task: {task}")


if __name__ == "__main__":
    main()
