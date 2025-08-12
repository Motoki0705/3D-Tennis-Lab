# filename: development/court_pose/01_vit_heatmap/lit_module.py
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from .model_components.decoder import UpsamplingDecoder
from .model_components.heatmap_head import HeatmapHead
from .model_components.vit_encoder import VitEncoder

class CourtLitModule(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # 1. モデルの構築
        self.encoder = VitEncoder(
            vit_name=self.config.model.vit_name,
            pretrained=self.config.model.pretrained
        )
        self.decoder = UpsamplingDecoder(
            in_channels=self.encoder.embed_dim,
            decoder_channels=self.config.model.decoder_channels,
            output_size=self.config.model.output_size
        )
        self.head = HeatmapHead(
            in_channels=self.config.model.decoder_channels[-1],
            num_keypoints=self.config.model.heatmap_channels
        )

        # 2. 損失関数
        self.loss_fn = nn.MSELoss()

        # 3. 評価指標 (PCK)
        self.pck_threshold = self.config.evaluation.pck_threshold

    def forward(self, x):
        features = self.encoder(x)
        upsampled = self.decoder(features)
        heatmaps = self.head(upsampled)
        return heatmaps

    def training_step(self, batch, batch_idx):
        images, target_heatmaps = batch
        pred_heatmaps = self(images)
        loss = self.loss_fn(pred_heatmaps, target_heatmaps)
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, target_heatmaps = batch
        pred_heatmaps = self(images)
        loss = self.loss_fn(pred_heatmaps, target_heatmaps)
        self.log('val/loss', loss, prog_bar=True, on_epoch=True)

        # PCKの計算
        pck = self.calculate_pck(pred_heatmaps, target_heatmaps)
        self.log(f'val/PCK@{self.pck_threshold}', pck, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.1, 
            patience=5,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": f"val/PCK@{self.pck_threshold}",
            },
        }

    def get_coords_from_heatmap(self, heatmap):
        """ヒートマップから座標を抽出 (argmax)"""
        batch_size, num_kps, h, w = heatmap.shape
        heatmap_reshaped = heatmap.reshape(batch_size, num_kps, -1)
        max_indices = torch.argmax(heatmap_reshaped, dim=2)
        
        ys = (max_indices / w).int().float()
        xs = (max_indices % w).int().float()
        
        # [B, K, 2] 形式の座標に
        coords = torch.stack([xs, ys], dim=2)
        return coords

    def calculate_pck(self, pred_heatmaps, target_heatmaps):
        """PCK (Percentage of Correct Keypoints) を計算"""
        pred_coords = self.get_coords_from_heatmap(pred_heatmaps)
        target_coords = self.get_coords_from_heatmap(target_heatmaps)

        # 閾値はヒートマップの対角線の長さに対する割合
        h, w = pred_heatmaps.shape[2:]
        threshold = self.pck_threshold * torch.tensor(h**2 + w**2).sqrt()

        # 座標間の距離を計算
        distances = torch.norm(pred_coords - target_coords, dim=2)
        
        # 正解ヒートマップにピークがあるキーポイントのみを評価対象とする
        # (v=0のキーポイントは除く)
        target_max_vals, _ = torch.max(target_heatmaps.reshape(target_heatmaps.shape[0], target_heatmaps.shape[1], -1), dim=2)
        mask = target_max_vals > 1e-4  # 小さな値はピークなしと見なす

        correct_predictions = (distances < threshold) * mask
        
        num_correct = correct_predictions.sum()
        num_visible_kps = mask.sum()
        
        pck = num_correct / num_visible_kps if num_visible_kps > 0 else torch.tensor(0.0)
        return pck.to(self.device)