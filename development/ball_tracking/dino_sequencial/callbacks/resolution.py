from __future__ import annotations

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer import Trainer


class ProgressiveResolutionCallback(Callback):
    """Synchronises progressive resolution schedule with the Lightning training loop."""

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:  # type: ignore[override]
        self._maybe_set_max_epochs(trainer)
        self._maybe_update_epoch(trainer)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:  # type: ignore[override]
        self._maybe_set_max_epochs(trainer)
        self._maybe_update_epoch(trainer)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:  # type: ignore[override]
        self._maybe_update_epoch(trainer)

    def _maybe_set_max_epochs(self, trainer: Trainer) -> None:
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return
        if hasattr(datamodule, "set_progressive_max_epochs"):
            max_epochs = getattr(trainer, "max_epochs", None)
            if max_epochs is not None:
                datamodule.set_progressive_max_epochs(int(max_epochs))

    def _maybe_update_epoch(self, trainer: Trainer) -> None:
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return
        if hasattr(datamodule, "on_progressive_epoch_start"):
            current_epoch = getattr(trainer, "current_epoch", 0)
            datamodule.on_progressive_epoch_start(int(current_epoch))


__all__ = ["ProgressiveResolutionCallback"]
