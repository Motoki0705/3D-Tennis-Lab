from __future__ import annotations

import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


class DetectionVisualizer(pl.Callback):
    """Callback to visualize model predictions during validation."""

    def __init__(self, num_samples: int = 4, score_threshold: float = 0.5):
        super().__init__()
        self.num_samples = num_samples
        self.score_threshold = score_threshold
        self.samples_logged = 0

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Reset the counter at the start of each validation epoch."""
        self.samples_logged = 0

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log a few images with predictions at the end of a validation batch."""
        if self.samples_logged >= self.num_samples:
            return

        logger = trainer.logger
        if not isinstance(logger, TensorBoardLogger):
            print(f"Warning: DetectionVisualizer requires TensorBoardLogger, but found {type(logger)}.")
            return

        # Extract data from outputs dict
        images = outputs.get("images")
        targets = outputs.get("targets")
        predictions = outputs.get("predictions")

        if not all([images, targets, predictions]):
            print(
                "Warning: DetectionVisualizer expects 'images', 'targets', and 'predictions' in validation_step output."
            )
            return

        for i in range(len(images)):
            if self.samples_logged >= self.num_samples:
                break

            img, gt, pred = images[i], targets[i], predictions[i]

            # Convert image to uint8 for drawing
            img_uint8 = (img * 255).to(torch.uint8)

            # Draw ground truth boxes (green)
            img_with_gt = torchvision.utils.draw_bounding_boxes(
                image=img_uint8,
                boxes=gt["boxes"],
                colors="green",
                width=2,
            )

            # Filter predictions by score and draw them (red)
            scores = pred["scores"]
            high_conf_mask = scores > self.score_threshold
            pred_boxes = pred["boxes"][high_conf_mask]
            pred_labels = [
                f"{label.item()}: {score:.2f}"
                for label, score in zip(pred["labels"][high_conf_mask], scores[high_conf_mask])
            ]

            img_with_preds = torchvision.utils.draw_bounding_boxes(
                image=img_uint8.clone(),  # Use a clone to draw on a fresh image
                boxes=pred_boxes,
                labels=pred_labels,
                colors="red",
                width=2,
            )

            # Create a side-by-side comparison
            # Ensure images are the same height before concatenating
            h = max(img_with_gt.shape[1], img_with_preds.shape[1])
            w = max(img_with_gt.shape[2], img_with_preds.shape[2])

            transform = torchvision.transforms.Resize((h, w))
            img_with_gt_resized = transform(img_with_gt)
            img_with_preds_resized = transform(img_with_preds)

            combined_img = torch.cat([img_with_gt_resized, img_with_preds_resized], dim=2)

            logger.experiment.add_image(
                f"validation/sample_{self.samples_logged}",
                combined_img,
                global_step=trainer.global_step,
            )
            self.samples_logged += 1
