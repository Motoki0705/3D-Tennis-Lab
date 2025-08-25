# filename: development/court_pose/01_vit_heatmap/lit_module.py
import torch

from ...utils.lightning.base_lit_module import BaseLitModule
from ...utils.loss import loss_registry
from .model.vit_heatmap import VitHeatmapModel


class CourtLitModule(BaseLitModule):
    def __init__(self, config):
        model = VitHeatmapModel(
            vit_name=config.model.vit_name,
            pretrained=config.model.pretrained,
            decoder_channels=config.model.decoder_channels,
            output_size=config.model.output_size,
            heatmap_channels=config.model.heatmap_channels,
        )

        loss_fn = self._get_loss_fn(config.training.loss)
        metric_fns = {
            f"PCK@{config.evaluation.pck_threshold}": lambda preds, targets: self.calculate_pck(
                preds, targets, threshold_ratio=config.evaluation.pck_threshold
            )
        }

        super().__init__(config=config, model=model, loss_fn=loss_fn, metric_fns=metric_fns)

        # FocalLossの場合はsigmoidを適用する必要がある
        self.needs_sigmoid = config.training.loss.name in ["focal"]

        self.freeze_vit_epochs = config.training.freeze_vit_epochs
        if self.freeze_vit_epochs > 0:
            self._freeze_vit()

    def _get_loss_fn(self, loss_config):
        loss_name = loss_config.name
        loss_params = loss_config.params if hasattr(loss_config, "params") else {}
        return loss_registry.get(loss_name, **loss_params)

    def _calculate_loss(self, preds, targets):
        # FocalLossはsigmoidをかけてから損失を計算
        if self.needs_sigmoid:
            preds = torch.sigmoid(preds)
        return self.loss_fn(preds, targets)

    def _freeze_vit(self):
        print("Freezing ViT encoder.")
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def _unfreeze_vit(self):
        print("Unfreezing ViT encoder.")
        for param in self.model.encoder.parameters():
            param.requires_grad = True

    def on_train_epoch_start(self):
        if self.current_epoch == self.freeze_vit_epochs:
            self._unfreeze_vit()
            print("ViT encoder unfrozen. Optimizer will now update its parameters.")

    def configure_optimizers(self):
        # パラメータをViTとそれ以外に分割
        vit_params = self.model.encoder.parameters()
        other_params = list(self.model.decoder.parameters()) + list(self.model.head.parameters())

        # パラメータグループごとに異なる学習率を設定
        param_groups = [
            {"params": other_params, "lr": self.config.training.lr},
            {"params": vit_params, "lr": self.config.training.vit_lr},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.training.weight_decay,
        )

        # BaseLitModuleと同様のスケジューラ設定
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",  # val_lossを監視するためmin
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

    # callback(HeatmapImageLogger)用の出力を追加
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)  # ロジット想定(argmax位置はsigmoid有無で不変)
        loss = self._calculate_loss(preds, targets)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)

        for name, fn in self.metric_fns.items():
            val = fn(preds, targets)
            self.log(f"val/{name}", val, prog_bar=True, on_epoch=True)

        # HeatmapImageLogger用の出力
        return {
            "loss": loss,
            "images": images.detach().cpu(),
            "pred_heatmaps": preds.detach().cpu(),  # 可視化時にsigmoidする場合は側で対応
            "target_heatmaps": targets.detach().cpu(),
        }

    def test_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)
        loss = self._calculate_loss(preds, targets)
        self.log("test/loss", loss, prog_bar=True, on_epoch=True)

        for name, fn in self.metric_fns.items():
            val = fn(preds, targets)
            self.log(f"test/{name}", val, prog_bar=True, on_epoch=True)

        # HeatmapImageLogger用の出力
        return {
            "loss": loss,
            "images": images.detach().cpu(),
            "pred_heatmaps": preds.detach().cpu(),
            "target_heatmaps": targets.detach().cpu(),
        }

    # ====== Heatmap → 座標変換 ======
    def get_coords_from_heatmap(self, heatmap):
        # 注意: argmaxは単調変換に不変なのでロジットのままでも座標は同一
        batch_size, num_kps, h, w = heatmap.shape
        heatmap_reshaped = heatmap.reshape(batch_size, num_kps, -1)
        max_indices = torch.argmax(heatmap_reshaped, dim=2)

        ys = (max_indices / w).int().float()
        xs = (max_indices % w).int().float()

        coords = torch.stack([xs, ys], dim=2)  # [B, K, 2]
        return coords

    # ====== PCK計算 ======
    def calculate_pck(self, pred_heatmaps, target_heatmaps, threshold_ratio=0.05):
        pred_coords = self.get_coords_from_heatmap(pred_heatmaps)
        target_coords = self.get_coords_from_heatmap(target_heatmaps)

        h, w = pred_heatmaps.shape[2:]
        threshold = threshold_ratio * torch.tensor(h**2 + w**2).sqrt()

        distances = torch.norm(pred_coords - target_coords, dim=2)

        target_max_vals, _ = torch.max(
            target_heatmaps.reshape(target_heatmaps.shape[0], target_heatmaps.shape[1], -1), dim=2
        )
        mask = target_max_vals > 1e-4

        correct_predictions = (distances < threshold) * mask

        num_correct = correct_predictions.sum()
        num_visible_kps = mask.sum()

        pck = num_correct / num_visible_kps if num_visible_kps > 0 else torch.tensor(0.0)
        return pck
