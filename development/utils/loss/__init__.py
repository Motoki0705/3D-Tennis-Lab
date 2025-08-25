# filename: development/utils/loss/__init__.py
import torch.nn as nn

# カスタム損失関数をインポートして登録
from .focal_loss import FocalLoss
from .kl_divergence_loss import KLDivLoss
from .loss_registry import loss_registry, register_loss

# PyTorchの標準損失関数を登録
loss_registry.register("mse", nn.MSELoss)
loss_registry.register("bce", nn.BCEWithLogitsLoss)

__all__ = ["loss_registry", "register_loss", "FocalLoss", "KLDivLoss"]
