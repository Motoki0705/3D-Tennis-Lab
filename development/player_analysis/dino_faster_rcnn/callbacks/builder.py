from typing import List
from omegaconf import DictConfig
from pytorch_lightning import Callback
import hydra


def build_callbacks(cfg: DictConfig) -> List[Callback]:
    """
    Instantiates a list of callbacks from the given config.
    """
    callbacks: List[Callback] = []
    if not cfg:
        return callbacks

    for _, cb_conf in cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks
