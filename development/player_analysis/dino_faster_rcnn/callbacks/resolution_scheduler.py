from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


@dataclass
class LinearMixSchedule:
    start_epoch: int = 0
    end_epoch: Optional[int] = None
    start_p_high: float = 0.0
    end_p_high: float = 1.0

    def value_at(self, epoch: int, max_epochs: int) -> float:
        end = self.end_epoch if self.end_epoch is not None else max_epochs - 1
        start = max(0, int(self.start_epoch))
        end = max(start, int(end))
        if epoch <= start:
            return float(self.start_p_high)
        if epoch >= end:
            return float(self.end_p_high)
        span = max(1, end - start)
        alpha = (epoch - start) / span
        return float((1 - alpha) * self.start_p_high + alpha * self.end_p_high)


class ResolutionMixScheduler(pl.Callback):
    """
    Updates the probability of sampling high-resolution images during training.

    Notes on dataloader reloads:
    - To ensure worker processes see updated probabilities, set
      `reload_dataloaders_every_n_epochs=1` in the Trainer config.
    - This callback updates the ratio at the end of each epoch so that the next
      epoch's DataLoader picks up the new setting when it reloads.
    """

    def __init__(
        self,
        start_epoch: int = 0,
        end_epoch: Optional[int] = None,
        start_p_high: float = 0.0,
        end_p_high: float = 1.0,
        log: bool = True,
    ) -> None:
        super().__init__()
        self.schedule = LinearMixSchedule(
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            start_p_high=start_p_high,
            end_p_high=end_p_high,
        )
        self.log = log

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        dm = trainer.datamodule
        if hasattr(dm, "set_resolution_mix_ratio"):
            # Initialize for epoch 0
            p0 = self.schedule.value_at(0, trainer.max_epochs or 1)
            dm.set_resolution_mix_ratio(p0)
            if self.log and isinstance(trainer.logger, TensorBoardLogger):
                # Logging via logger.experiment is allowed in callbacks at this stage
                trainer.logger.experiment.add_scalar("mix/p_high", p0, global_step=0)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        next_epoch = trainer.current_epoch + 1
        dm = trainer.datamodule
        if hasattr(dm, "set_resolution_mix_ratio"):
            p_next = self.schedule.value_at(next_epoch, trainer.max_epochs or (next_epoch + 1))
            dm.set_resolution_mix_ratio(p_next)
            if self.log and isinstance(trainer.logger, TensorBoardLogger):
                # Log the next epoch's ratio using epoch index as step for clarity
                trainer.logger.experiment.add_scalar("mix/p_high_next", p_next, global_step=next_epoch)
