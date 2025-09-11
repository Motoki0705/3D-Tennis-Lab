from __future__ import annotations

import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger

from ..callbacks.builder import build_callbacks
from ..training.datamodule import DetectionDataModule
from ..training.lit_module import DetectionLitModule
from ..model.dino_faster_rcnn import DinoFasterRCNN


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
            image_size_low=self.cfg.data.get("image_size_low", None),
            image_size_high=self.cfg.data.get("image_size_high", None),
        )

        # Model
        model = DinoFasterRCNN(**self.cfg.model)
        lit = DetectionLitModule(model=model, lr=self.cfg.lit_module.lr, weight_decay=self.cfg.lit_module.weight_decay)

        # Instantiate callbacks from our build system
        callbacks = build_callbacks(self.cfg.callbacks)

        logger = TensorBoardLogger(**self.cfg.logger)

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
            logger=logger,
        )

        trainer.fit(lit, datamodule=dm)
