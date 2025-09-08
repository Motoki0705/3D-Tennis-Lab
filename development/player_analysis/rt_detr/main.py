import logging
from typing import Literal

import hydra
from omegaconf import DictConfig
import torch

torch.set_float32_matmul_precision("medium")
logger = logging.getLogger(__name__)


def _dispatch(task: Literal["train", "infer"], cfg: DictConfig):
    if task == "train":
        from .runner.train import TrainRunner

        TrainRunner(cfg).run()
    elif task == "infer":
        # Optional: add inference runner later
        raise SystemExit("infer タスクは未実装です。train を使用してください。")
    else:
        raise ValueError(f"Unknown task: {task}")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    task = cfg.get("task", "train")
    logger.info(f"Running task: {task}")
    _dispatch(task, cfg)


if __name__ == "__main__":
    main()
