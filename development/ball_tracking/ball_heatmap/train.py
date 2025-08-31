import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from .datamodule import BallDataModule
from .heatmap_logger_v2 import HeatmapLoggerV2
from .lit_module import BallLitModule


@hydra.main(config_path="configs", config_name="ball_heatmap_v1", version_base=None)
def train(cfg: DictConfig):
    torch.set_float32_matmul_precision("high")

    # Loggers and callbacks
    logger = TensorBoardLogger("tb_logs", name="ball_heatmap_v1")

    # ModelCheckpoint/EarlyStopping config shortcuts
    monitor = "val/total_loss"
    mode = "min"
    checkpoint = ModelCheckpoint(
        monitor=monitor, mode=mode, save_top_k=1, filename="epoch={epoch}-vloss={val/total_loss:.4f}"
    )
    early_stop = EarlyStopping(monitor=monitor, mode=mode, patience=5, verbose=True)
    lrmon = LearningRateMonitor(logging_interval="epoch")
    progress = RichProgressBar()
    hlogger = HeatmapLoggerV2(num_samples=int(cfg.callbacks.heatmap_logger.num_samples))

    # Data + Model
    dm = BallDataModule(cfg)
    model = BallLitModule(cfg)

    trainer = pl.Trainer(
        max_epochs=int(cfg.training.max_epochs),
        accelerator=str(cfg.training.accelerator),
        devices=int(cfg.training.devices),
        precision=str(cfg.training.precision),
        logger=logger,
        callbacks=[checkpoint, early_stop, lrmon, progress, hlogger],
        reload_dataloaders_every_n_epochs=1,  # allow epoch-driven sampler updates
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    train()
