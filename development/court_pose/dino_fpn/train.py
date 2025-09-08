# Train script for DINO-FPN heatmap model (Hydra-based)
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from ...utils.callbacks.heatmap_logger import HeatmapImageLogger
from ...utils.transformers.keypoint_transformer import prepare_transforms
from .datamodule import CourtDataModule
from .lit_module import DinoFpnLitModule


@hydra.main(config_path="configs", config_name="dino_fpn_v1", version_base=None)
def train(config: DictConfig):
    torch.set_float32_matmul_precision("high")

    # Logger and callbacks
    logger = TensorBoardLogger("tb_logs", name="dino_fpn_v1")
    checkpoint_callback = ModelCheckpoint(
        monitor=config.callbacks.checkpoint.monitor,
        mode=config.callbacks.checkpoint.mode,
        save_top_k=config.callbacks.checkpoint.save_top_k,
        filename="epoch={epoch}-val_loss={val/loss:.4f}",
        auto_insert_metric_name=False,
    )
    early_stop_callback = EarlyStopping(
        monitor=config.callbacks.early_stopping.monitor,
        patience=config.callbacks.early_stopping.patience,
        mode=config.callbacks.early_stopping.mode,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = RichProgressBar()
    heatmap_logger = HeatmapImageLogger(num_samples=config.callbacks.heatmap_logger.num_samples)

    # Data and model
    train_t, val_t = prepare_transforms(img_size=config.dataset.img_size)
    datamodule = CourtDataModule(config=config, train_transforms=train_t, val_transforms=val_t, test_transforms=val_t)
    model = DinoFpnLitModule(config)

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        precision=config.training.precision,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, progress_bar, heatmap_logger],
    )
    trainer.fit(model, datamodule=datamodule)
    print("--- Starting Test ---")
    trainer.test(datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    train()
