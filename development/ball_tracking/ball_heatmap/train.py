import hydra
import os
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from development.ball_tracking.ball_heatmap.callbacks.sequence_debug_logger import SequenceDebugLogger
from development.ball_tracking.ball_heatmap.training.datamodule import BallDataModule
from development.ball_tracking.ball_heatmap.training.module import BallLightningModule
from development.ball_tracking.ball_heatmap.training.registry import (
    build_adversary,
    build_semisup_strategy,
)

torch.set_float32_matmul_precision("medium")  # or 'high'


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """
    Main training script for the ball heatmap model.
    Uses Hydra for configuration management and PyTorch Lightning for training.
    """
    # For reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # --- Initialize DataModule ---
    datamodule = BallDataModule(cfg)

    # --- Initialize Model ---
    # Build strategies from the registry based on the config
    semisup_strategy = build_semisup_strategy(cfg)
    adversary = build_adversary(cfg)
    model = BallLightningModule(cfg, semisup_strategy=semisup_strategy, adversary=adversary)

    # --- Initialize Logger ---
    logger = TensorBoardLogger("tb_logs", name="ball_heatmap_logs")
    # Save checkpoints under the same hierarchy as TensorBoard logs, split by version
    ckpt_dir = os.path.join(logger.log_dir, "checkpoints")

    # --- Initialize Callbacks ---
    callbacks = []
    # Callback to save the best models based on validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor="val/total",
        dirpath=ckpt_dir,
        filename=f"{cfg.semisup.name}-ball-heatmap-{{epoch:02d}}-{{val/total:.2f}}",
        save_top_k=3,
        mode="min",
    )
    callbacks.append(checkpoint_callback)

    # Callback to log learning rates
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Visualization callbacks for debugging and analysis
    callbacks.append(SequenceDebugLogger(num_samples=4))

    # --- Initialize Trainer ---
    # Instantiate the Trainer using the configuration, and add callbacks and logger
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # --- Start Training ---
    trainer.fit(model, datamodule=datamodule)

    # --- Optional: Start Testing ---
    # After training, you can run the test set
    # trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    train()
