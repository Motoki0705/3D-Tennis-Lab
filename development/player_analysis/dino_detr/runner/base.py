from __future__ import annotations

import logging
from abc import ABC, abstractmethod

try:
    import torch
except ModuleNotFoundError:
    raise SystemExit("PyTorch (torch) が見つかりません。環境を有効化してください。")


logger = logging.getLogger(__name__)


class BaseRunner(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = self._get_device(getattr(cfg, "device", "auto"))

    @staticmethod
    def _get_device(name: str) -> torch.device:
        if name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(name)

    @abstractmethod
    def run(self):
        raise NotImplementedError
