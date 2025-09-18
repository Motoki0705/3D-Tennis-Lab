from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from pytorch_lightning.callbacks import Callback
from torchvision.utils import make_grid

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class SequenceHeatmapVisualizationCallback(Callback):
    """Log input/prediction/target heatmaps for sequences to TensorBoard during validation."""

    def __init__(self, *, max_sequences: int, log_every_n_epochs: int) -> None:
        super().__init__()
        # This parameter is kept for config compatibility, but we only visualize the first sequence.
        self.max_sequences = max(1, int(max_sequences))
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

        # Batch is (img_seq, target_heatmap_seq)
        # img_seq is (B, T, C, H, W)
        inputs, targets = batch
        inputs = inputs.to(pl_module.device)
        targets = targets.to(pl_module.device)

        with torch.no_grad():
            preds = pl_module(inputs)

        # Visualize the first sequence of the batch
        inputs_vis = inputs[0].detach().cpu()  # (T, C, H, W)
        preds_vis = preds[0].detach().cpu()  # (T, 1, H_out, W_out)
        targets_vis = targets[0].detach().cpu()  # (T, 1, H_out, W_out)

        inputs_vis = _denormalize(inputs_vis)

        # Convert heatmaps to RGB for visualization
        preds_vis_rgb = preds_vis.repeat(1, 3, 1, 1)
        targets_vis_rgb = targets_vis.repeat(1, 3, 1, 1)

        sequence_length = inputs.shape[1]
        step = trainer.global_step

        for logger in trainer.loggers:
            _log_image(logger, "val/input_sequence", make_grid(inputs_vis, nrow=sequence_length), step)
            _log_image(
                logger,
                "val/pred_heatmap_sequence",
                make_grid(preds_vis_rgb, nrow=sequence_length, normalize=True),
                step,
            )
            _log_image(
                logger,
                "val/target_heatmap_sequence",
                make_grid(targets_vis_rgb, nrow=sequence_length, normalize=True),
                step,
            )


def _denormalize(tensor: torch.Tensor) -> torch.Tensor:
    # Denormalize a batch of images
    mean = _IMAGENET_MEAN.to(tensor.device)
    std = _IMAGENET_STD.to(tensor.device)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)


def _first_batch(val_loader: Any) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if val_loader is None:
        return None
    try:
        return next(iter(val_loader))
    except StopIteration:
        return None


def _log_image(logger: Any, tag: str, image: torch.Tensor, step: int) -> None:
    experiment = getattr(logger, "experiment", None)
    if experiment is None or not hasattr(experiment, "add_image"):
        return
    experiment.add_image(tag, image, global_step=step)


__all__ = ["SequenceHeatmapVisualizationCallback"]
