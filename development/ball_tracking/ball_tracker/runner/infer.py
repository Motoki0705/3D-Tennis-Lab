from __future__ import annotations

import glob
import json
import logging
import os

import pytorch_lightning as pl
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from tqdm import tqdm

from ..training.datamodule import BallDataModule, DataModuleConfig
from ..training.lit_module import SequenceLitModule

logger = logging.getLogger(__name__)


class InferRunner:
    """Runs inference on a trained model."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def run(self):
        pl.seed_everything(self.cfg.data.split_seed, workers=True)

        # 1. --- DataModule ---
        dm_cfg = DataModuleConfig(
            labeled_json=abspath(self.cfg.data.labeled_json),
            sequence_length=int(self.cfg.data.sequence_length),
            predict_offset=int(self.cfg.data.predict_offset),
            val_ratio=float(self.cfg.data.val_ratio),
            split_seed=int(self.cfg.data.split_seed),
            batch_size=int(self.cfg.data.batch_size),
            num_workers=int(self.cfg.data.num_workers),
        )
        datamodule = BallDataModule(dm_cfg)
        datamodule.setup("fit")  # Use fit to have train/val splits

        # Select the dataloader based on config
        split = self.cfg.inference.get("dataset_split", "val")
        if split == "val":
            predict_loader = datamodule.val_dataloader()
        elif split == "train":
            predict_loader = datamodule.train_dataloader()
        elif split == "all":
            # This requires creating a new dataloader with the full dataset
            datamodule.setup(stage=None)
            predict_loader = datamodule.val_dataloader()  # Re-using val_dataloader logic for the full set
        else:
            raise ValueError(f"Invalid dataset_split: {split}. Choose from 'val', 'train', 'all'.")
        logger.info(f"Using '{split}' dataset for inference.")

        # 2. --- Load Model from Checkpoint ---
        ckpt_path = self._get_checkpoint_path()
        logger.info(f"Loading model from checkpoint: {ckpt_path}")
        lit_module = SequenceLitModule.load_from_checkpoint(checkpoint_path=ckpt_path)

        # 3. --- Run Prediction ---
        trainer = pl.Trainer(
            accelerator=self.cfg.training.accelerator,
            devices=self.cfg.training.devices,
            precision=self.cfg.training.precision,
            logger=False,  # No logging during inference
        )

        logger.info("Starting inference...")
        predictions_batched = trainer.predict(lit_module, dataloaders=predict_loader)

        # 4. --- Process and Save Results ---
        if not predictions_batched:
            logger.warning("Inference produced no results.")
            return

        # Flatten the list of lists of dictionaries
        results = []
        limit = self.cfg.inference.get("limit_samples")
        for batch in tqdm(predictions_batched, desc="Processing results"):
            for i in range(len(batch["prediction"])):
                if limit is not None and len(results) >= limit:
                    break
                results.append({
                    "input_sequence": batch["input_sequence"][i],
                    "ground_truth": batch["ground_truth"][i],
                    "prediction": batch["prediction"][i],
                })
            if limit is not None and len(results) >= limit:
                break

        output_path = abspath(self.cfg.inference.output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info(f"Saving {len(results)} inference results to {output_path}")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("Inference finished.")

    def _get_checkpoint_path(self) -> str:
        path = self.cfg.inference.checkpoint_path
        if path == "best":
            # Find the best checkpoint based on the monitor metric in the latest experiment
            log_dir = abspath(f"tb_logs/{self.cfg.experiment_name}")
            # Find the latest version
            versions = sorted(
                [
                    d
                    for d in os.listdir(log_dir)
                    if os.path.isdir(os.path.join(log_dir, d)) and d.startswith("version_")
                ],
                reverse=True,
            )
            if not versions:
                raise FileNotFoundError(f"No experiment versions found in {log_dir}")
            latest_version_dir = os.path.join(log_dir, versions[0], "checkpoints")

            # Find the checkpoint file with the best score in its name
            checkpoints = glob.glob(os.path.join(latest_version_dir, "*.ckpt"))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {latest_version_dir}")

            # A simple heuristic: find the one with the lowest "valloss" in the name
            best_ckpt = min(checkpoints, key=lambda p: float(p.split("valloss=")[-1].replace(".ckpt", "")))
            return best_ckpt
        else:
            return abspath(path)
