from __future__ import annotations

import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger

from ..callbacks.builder import build_callbacks
from ..training.datamodule import DetectionDataModule
from ..training.lit_module import DetectionLitModule
from ..model.dino_faster_rcnn import DinoFasterRCNN
from ..utils.model_io import load_checkpoint_into_model


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
            aspect_ratio=self.cfg.data.get("aspect_ratio", None),
            target_category=self.cfg.data.get("target_category", "player"),
            required_instances_per_image=self.cfg.data.get("required_instances_per_image", 2),
            splits=self.cfg.data.get("splits", None),
        )

        # Model
        model = DinoFasterRCNN(**self.cfg.model)
        # Optionally initialize from a checkpoint (weights only)
        ckpt_path = self.cfg.get("ckpt_path", None)
        if ckpt_path:
            load_checkpoint_into_model(model, ckpt_path)
        # Optimizer/Scheduler config come from training config group
        optim_cfg = self.cfg.training.get("optimizer", {}) if hasattr(self.cfg, "training") else {}
        sched_cfg = self.cfg.training.get("lr_scheduler", {}) if hasattr(self.cfg, "training") else {}
        # Backward compatibility: allow lr/weight_decay in lit_module if present
        lr = optim_cfg.get("lr", self.cfg.get("lit_module", {}).get("lr", 1e-4))
        weight_decay = optim_cfg.get("weight_decay", self.cfg.get("lit_module", {}).get("weight_decay", 1e-4))

        lit = DetectionLitModule(
            model=model,
            lr=lr,
            weight_decay=weight_decay,
            optimizer_cfg=optim_cfg,
            lr_scheduler_cfg=sched_cfg,
        )

        # Instantiate callbacks from our build system
        callbacks = build_callbacks(self.cfg.callbacks)

        logger = TensorBoardLogger(**self.cfg.logger)

        tr = self.cfg.training
        trainer = pl.Trainer(
            accelerator=tr.accelerator,
            devices=tr.devices,
            precision=tr.precision,
            max_epochs=tr.max_epochs,
            default_root_dir=getattr(tr, "default_root_dir", None),
            log_every_n_steps=tr.log_every_n_steps,
            val_check_interval=tr.val_check_interval,
            num_sanity_val_steps=tr.num_sanity_val_steps,
            gradient_clip_val=tr.gradient_clip_val,
            detect_anomaly=tr.detect_anomaly,
            callbacks=callbacks,
            logger=logger,
            reload_dataloaders_every_n_epochs=getattr(tr, "reload_dataloaders_every_n_epochs", 0),
        )

        trainer.fit(lit, datamodule=dm)
