# filename: base_module.py
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


class BaseLitModule(LightningModule):
    """
    汎用的な LightningModule ベースクラス
    モデル, 損失関数, 評価関数(PCKなど)を引数で注入して利用可能。
    """

    def __init__(
        self,
        config,
        model: nn.Module,
        loss_fn: nn.Module,
        metric_fns: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        self.model = model
        self.loss_fn = loss_fn
        self.metric_fns = metric_fns or {}

    # ====== Forward ======
    def forward(self, x):
        return self.model(x)

    # ====== Train step ======
    def training_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)
        loss = self.loss_fn(preds, targets)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # ====== Validation step ======
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)
        loss = self.loss_fn(preds, targets)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)

        for name, fn in self.metric_fns.items():
            val = fn(preds, targets)
            self.log(f"val/{name}", val, prog_bar=True, on_epoch=True)

        return loss

    # ====== Test step ======
    def test_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)
        loss = self.loss_fn(preds, targets)
        self.log("test/loss", loss, prog_bar=True, on_epoch=True)

        for name, fn in self.metric_fns.items():
            val = fn(preds, targets)
            self.log(f"test/{name}", val, prog_bar=True, on_epoch=True)

        return loss

    # ====== Optimizer設定 ======
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.1,
            patience=5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }
