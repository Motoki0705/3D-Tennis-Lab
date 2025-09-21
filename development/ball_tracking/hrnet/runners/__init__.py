import logging
from omegaconf import DictConfig

from .detect import DetectRunner

log = logging.getLogger(__name__)

__runner_factory = {
    "detect": DetectRunner,
}


def select_runner(cfg: DictConfig):
    runner_name = cfg["runner"]["name"]
    if not runner_name in __runner_factory.keys():
        raise KeyError("unknown runner: {}".format(runner_name))
    return __runner_factory[runner_name](cfg)
