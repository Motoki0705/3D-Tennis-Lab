import torch

from ...utils.lightning.base_lit_module import BaseLitModule
from ...utils.loss import loss_registry
from .model.model import DinoVitHeatmap


class DinoFpnLitModule(BaseLitModule):
    def __init__(self, config):
        model = DinoVitHeatmap(
            num_keypoints=config.model.heatmap_channels,
            decoder_channels=getattr(config.model, "decoder_channels", None),
            backbone_name=config.model.backbone_name,
            weights_path=config.model.weights_path,
        )

        loss_fn = self._get_loss_fn(config.training.loss)
        super().__init__(config=config, model=model, loss_fn=loss_fn, metric_fns={})

        # Freeze backbone by default (user requested)
        freeze_flag = bool(getattr(config.training, "freeze_backbone", True))
        self._set_backbone_requires_grad(not freeze_flag)

        # For focal loss, we need sigmoid before loss
        loss_name = getattr(config.training.loss, "name", "mse")
        self.needs_sigmoid = str(loss_name).lower() in ["focal"]

    def _get_loss_fn(self, loss_config):
        loss_name = loss_config.name
        loss_params = loss_config.params if hasattr(loss_config, "params") else {}
        return loss_registry.get(loss_name, **loss_params)

    def _set_backbone_requires_grad(self, requires_grad: bool):
        for p in self.model.backbone.parameters():
            p.requires_grad = requires_grad

    def configure_optimizers(self):
        # Two param groups: backbone (smaller LR), others (base LR). Ignore frozen params automatically.
        bb_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        other_params = [
            p for n, p in self.model.named_parameters() if not n.startswith("backbone.") and p.requires_grad
        ]
        param_groups = []
        if other_params:
            param_groups.append({"params": other_params, "lr": self.config.training.lr})
        if bb_params:
            param_groups.append({"params": bb_params, "lr": self.config.training.backbone_lr})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.config.training.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }

    # Override to support HeatmapImageLogger outputs
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)
        if self.needs_sigmoid:
            preds_loss = torch.sigmoid(preds)
        else:
            preds_loss = preds
        loss = self.loss_fn(preds_loss, targets)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        return {
            "loss": loss,
            "images": images.detach().cpu(),
            "pred_heatmaps": preds.detach().cpu(),
            "target_heatmaps": targets.detach().cpu(),
        }

    def test_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)
        if self.needs_sigmoid:
            preds_loss = torch.sigmoid(preds)
        else:
            preds_loss = preds
        loss = self.loss_fn(preds_loss, targets)
        self.log("test/loss", loss, prog_bar=True, on_epoch=True)
        return {
            "loss": loss,
            "images": images.detach().cpu(),
            "pred_heatmaps": preds.detach().cpu(),
            "target_heatmaps": targets.detach().cpu(),
        }
