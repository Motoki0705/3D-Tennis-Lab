from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from pytorch_lightning.callbacks import Callback
from torchvision.utils import make_grid

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class HeatmapVisualizationCallback(Callback):
    """Log input/prediction/target heatmaps to TensorBoard during validation."""

    def __init__(self, *, max_images: int, log_every_n_epochs: int) -> None:
        super().__init__()
        self.max_images = max(1, int(max_images))
        self.log_every_n_epochs = max(1, int(log_every_n_epochs))

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # pragma: no cover - requires trainer
        if not trainer.is_global_zero:
            return
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return
        if not trainer.loggers:
            return
        datamodule = trainer.datamodule
        if datamodule is None or not hasattr(datamodule, "val_dataloader"):
            return

        val_loader = datamodule.val_dataloader()
        batch = _first_batch(val_loader)
        if batch is None:
            return

        inputs, targets = batch
        inputs = inputs[: self.max_images].to(pl_module.device)
        targets = targets[: self.max_images].to(pl_module.device)

        with torch.no_grad():
            logits = pl_module(inputs)
            preds = torch.sigmoid(logits)

        inputs_cpu = inputs.detach().cpu()
        preds_cpu = preds.detach().cpu()
        targets_cpu = targets.detach().cpu()

        inputs_vis = _denormalize(inputs_cpu)
        preds_vis = preds_cpu.repeat(1, 3, 1, 1)
        targets_vis = targets_cpu.repeat(1, 3, 1, 1)

        nrow = min(self.max_images, 4)
        step = trainer.global_step

        for logger in trainer.loggers:
            _log_image(logger, "val/input", make_grid(inputs_vis, nrow=nrow), step)
            _log_image(logger, "val/pred_heatmap", make_grid(preds_vis, nrow=nrow, normalize=True), step)
            _log_image(logger, "val/target_heatmap", make_grid(targets_vis, nrow=nrow, normalize=True), step)


def _denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp(tensor * _IMAGENET_STD + _IMAGENET_MEAN, 0.0, 1.0)


def _first_batch(val_loader: Any) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if val_loader is None:
        return None
    if isinstance(val_loader, list):
        for loader in val_loader:
            batch = _first_batch(loader)
            if batch is not None:
                return batch
        return None
    try:
        iterator = iter(val_loader)
    except TypeError:
        return None
    try:
        return next(iterator)
    except StopIteration:
        return None


def _log_image(logger: Any, tag: str, image: torch.Tensor, step: int) -> None:
    experiment = getattr(logger, "experiment", None)
    if experiment is None or not hasattr(experiment, "add_image"):
        return
    experiment.add_image(tag, image, global_step=step)


__all__ = ["HeatmapVisualizationCallback"]
