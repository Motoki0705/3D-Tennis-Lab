from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from ..training.lit_module import SequenceLitModule


def _get_tensorboard_logger(trainer: Trainer) -> TensorBoardLogger | None:
    for logger in trainer.loggers:
        if isinstance(logger, TensorBoardLogger):
            return logger
    return None


class VisualizationCallback(Callback):
    """Callback to visualize model predictions during validation."""

    def __init__(self, plot_interval: int = 10, plot_samples: int = 5):
        super().__init__()
        self.plot_interval = plot_interval
        self.plot_samples = plot_samples

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: SequenceLitModule,
        outputs: torch.Tensor,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        # Plot only for the first batch and at specified epoch intervals
        if batch_idx != 0 or (pl_module.current_epoch + 1) % self.plot_interval != 0:
            return

        logger = _get_tensorboard_logger(trainer)
        if not logger:
            return

        # Extract data from batch
        input_seq, target_vec = batch
        input_seq = input_seq[: self.plot_samples]
        target_vec = target_vec[: self.plot_samples]

        # Get model prediction
        pred_vec, _, _ = pl_module._prepare(input_seq, target_vec)

        # Denormalize only the prediction vector, as it's in the normalized space.
        # input_seq and target_vec are from the dataloader and already in the original scale.
        if pl_module.standardizer and pl_module.standardizer.has_stats:
            pred_vec = pl_module.standardizer.denormalize(pred_vec)

        # Move data to CPU for plotting
        input_seq = input_seq.cpu().numpy()
        target_vec = target_vec.cpu().numpy()
        pred_vec = pred_vec.detach().cpu().numpy()

        for i in range(input_seq.shape[0]):
            fig = self._create_plot(input_seq[i], target_vec[i], pred_vec[i])
            tag = f"validation/prediction_sample_{i}"
            logger.experiment.add_figure(tag, fig, global_step=pl_module.current_epoch)
            plt.close(fig)

    def _create_plot(self, input_seq: torch.Tensor, target_vec: torch.Tensor, pred_vec: torch.Tensor) -> Figure:
        fig, ax = plt.subplots(figsize=(8, 8))
        # Plot input sequence trajectory (x, y)
        ax.plot(input_seq[:, 0], input_seq[:, 1], "o-", label="Input Sequence", color="blue")
        # Plot ground truth target
        ax.plot(target_vec[0], target_vec[1], "o", markersize=10, label="Ground Truth", color="green")
        # Plot predicted target
        ax.plot(pred_vec[0], pred_vec[1], "x", markersize=10, label="Prediction", color="red")

        # Connect last input point to target and prediction for clarity
        ax.plot([input_seq[-1, 0], target_vec[0]], [input_seq[-1, 1], target_vec[1]], "--", color="green", alpha=0.7)
        ax.plot([input_seq[-1, 0], pred_vec[0]], [input_seq[-1, 1], pred_vec[1]], "--", color="red", alpha=0.7)

        ax.set_title("Ball Trajectory Prediction")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.legend()
        ax.grid(True)
        ax.axis("equal")
        fig.tight_layout()
        return fig
