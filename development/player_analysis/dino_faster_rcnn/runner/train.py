from __future__ import annotations


import pytorch_lightning as pl
from omegaconf import DictConfig

from training.datamodule import DetectionDataModule
from training.lit_module import DetectionLitModule
from model.dino_faster_rcnn import DinoFasterRCNN


class TrainRunner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def run(self):
        pl.seed_everything(self.cfg.get("seed", 42), workers=True)

        # Data
        dm = DetectionDataModule(
            train_images=self.cfg.data.train.images,
            train_ann=self.cfg.data.train.ann,
            val_images=self.cfg.data.val.images,
            val_ann=self.cfg.data.val.ann,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            image_size=self.cfg.data.image_size,
        )

        # Model
        model = DinoFasterRCNN(**self.cfg.model)
        lit = DetectionLitModule(model=model)

        # Callbacks and logging
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                dirpath=self.cfg.trainer.checkpoint_dir,
                filename="epoch={epoch}-val_loss={val/loss:.3f}",
                monitor="val/loss",
                mode="min",
                save_top_k=3,
                save_last=True,
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
        ]

        trainer = pl.Trainer(
            accelerator=self.cfg.trainer.accelerator,
            devices=self.cfg.trainer.devices,
            precision=self.cfg.trainer.precision,
            max_epochs=self.cfg.trainer.max_epochs,
            default_root_dir=self.cfg.trainer.default_root_dir,
            log_every_n_steps=self.cfg.trainer.log_every_n_steps,
            val_check_interval=self.cfg.trainer.val_check_interval,
            num_sanity_val_steps=self.cfg.trainer.num_sanity_val_steps,
            gradient_clip_val=self.cfg.trainer.gradient_clip_val,
            detect_anomaly=self.cfg.trainer.detect_anomaly,
            callbacks=callbacks,
        )

        trainer.fit(lit, datamodule=dm)
