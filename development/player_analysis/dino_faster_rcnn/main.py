import os
from typing import Literal

import hydra
from omegaconf import DictConfig, OmegaConf

from runner.train import TrainRunner
from runner.infer import InferRunner


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    task: Literal["train", "infer"] = cfg.get("task", "train")
    print("Config:\n" + OmegaConf.to_yaml(cfg))

    if task == "train":
        TrainRunner(cfg).run()
    elif task == "infer":
        InferRunner(cfg).run()
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    os.environ.setdefault("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
    main()
