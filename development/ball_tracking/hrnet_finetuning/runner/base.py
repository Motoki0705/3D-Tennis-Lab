from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict

try:
    import torch
except ModuleNotFoundError:
    raise SystemExit("PyTorch (torch) が見つかりません。環境を有効化してください。")


logger = logging.getLogger(__name__)


class BaseRunner(ABC):
    """Unified interface for train/infer runners.

    Subclasses must implement `run(self)` and can rely on common utilities here.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = self._get_device(getattr(cfg, "device", "auto"))

    @staticmethod
    def _get_device(name: str) -> torch.device:
        if name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(name)

    @staticmethod
    def _load_state_dict(obj) -> Dict[str, torch.Tensor]:
        # Support common wrappers
        if isinstance(obj, dict):
            for k in ["state_dict", "model_state_dict", "model_state", "model", "weights"]:
                if k in obj and isinstance(obj[k], dict):
                    return obj[k]
            # plain state_dict
            if any(isinstance(v, torch.Tensor) for v in obj.values()):
                return obj
        raise ValueError("Unsupported checkpoint format: state_dict not found")

    @abstractmethod
    def run(self):
        raise NotImplementedError
