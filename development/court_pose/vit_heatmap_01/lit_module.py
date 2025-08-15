# filename: development/court_pose/01_vit_heatmap/lit_module.py
import torch
import torch.nn as nn

from ...utils.lightning.base_lit_module import BaseLitModule
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
        loss_fn = nn.MSELoss()

        # metric関数群
        metric_fns = {
            f"PCK@{config.evaluation.pck_threshold}": lambda preds, targets: self.calculate_pck(
                preds, targets, threshold_ratio=config.evaluation.pck_threshold
            )
        }

        super().__init__(config=config, model=model, loss_fn=loss_fn, metric_fns=metric_fns)

    # ====== Heatmap → 座標変換 ======
    def get_coords_from_heatmap(self, heatmap):
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
