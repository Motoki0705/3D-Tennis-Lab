# development/ball_tracking/ball_heatmap/sequence_debug_logger.py
import io
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_lightning import Callback, LightningModule, Trainer
from torchvision import transforms
from torchvision.utils import make_grid

from development.ball_tracking.ball_heatmap.trackers.features import soft_argmax_2d


class SequenceDebugLogger(Callback):
    def __init__(self, num_samples: int = 4, draw_multiscale: bool = True):
        super().__init__()
        self.num_samples = num_samples
        self.draw_multiscale = draw_multiscale
        self._buffer = None

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx):
        if batch_idx > 0:
            return
        # Keep CPU tensors as prepared by validation_step, and augment with GT info from batch
        buf = dict(outputs)
        try:
            sup = batch.get("sup", {})
            targets = sup.get("targets", {})
            # These come batched from collate.py
            buf["vis_state"] = targets.get("vis_state").detach().cpu()
            buf["speed_gt_norm"] = targets.get("speed").detach().cpu()  # normalized by [W,H]
        except Exception:
            pass
        self._buffer = buf

    def _draw_trajectory_on_grid(
        self, images: torch.Tensor, pred_xy: torch.Tensor, gt_xy: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        B, T, C, H, W = images.shape
        vis_images = []
        for i in range(min(B, self.num_samples)):
            # Use the center frame as the background for the trajectory plot
            center_frame_pil = transforms.ToPILImage()(images[i, T // 2])

            fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
            ax.imshow(center_frame_pil)
            ax.axis("off")

            # Draw predicted trajectory (velocity-colored)
            points = pred_xy[i].numpy()
            velocities = np.linalg.norm(np.diff(points, axis=0), axis=1)
            velocities = np.insert(velocities, 0, 0)
            ax.plot(points[:, 0], points[:, 1], "r-", label="Predicted", alpha=0.7)
            ax.scatter(points[:, 0], points[:, 1], c=velocities, cmap="viridis", s=10)

            # Draw ground truth trajectory
            valid_mask = valid[i].bool()
            if valid_mask.any():
                gt_points = gt_xy[i][valid_mask].numpy()
                if len(gt_points) > 0:
                    ax.plot(gt_points[:, 0], gt_points[:, 1], "g--", label="Ground Truth", alpha=0.7)

            ax.legend()
            fig.tight_layout(pad=0)

            with io.BytesIO() as buf:
                fig.savefig(buf, format="png")
                buf.seek(0)
                img = Image.open(buf).convert("RGB")
                vis_images.append(transforms.ToTensor()(img))
            plt.close(fig)

        return make_grid(vis_images, nrow=1)

    def _hm_to_bt(self, hm: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """Ensure heatmap shape is [B,T,1,Hs,Ws]. Accepts [B,T,1,Hs,Ws] or [B*T,1,Hs,Ws]."""
        if hm.dim() == 5:
            return hm
        if hm.dim() == 4 and hm.shape[0] == B * T:
            return hm.view(B, T, *hm.shape[1:])
        raise ValueError(f"Unexpected heatmap shape: {hm.shape}")

    def _pick_highest_res(self, hmaps: List[torch.Tensor], strides: List[int]) -> Tuple[torch.Tensor, int, int, int]:
        """Return highest-resolution heatmap [B,T,1,Hs,Ws] and its stride, Hs, Ws.
        Assumes smaller stride => higher resolution.
        """
        if not hmaps:
            raise ValueError("Empty heatmap list")
        # Choose index with smallest stride if provided; else default to 0
        if strides and len(strides) == len(hmaps):
            idx = int(min(range(len(strides)), key=lambda i: strides[i]))
        else:
            idx = 0
        hm = hmaps[idx]
        return hm, int(strides[idx] if strides else 1), hm.shape[-2], hm.shape[-1]

    def _draw_sequence_points_grid(
        self,
        images: torch.Tensor,
        points: torch.Tensor,
        speed_px: Optional[torch.Tensor],
        vis_state: Optional[torch.Tensor],
        title: str,
    ) -> torch.Tensor:
        """Draw per-frame points and optional speed vectors on a grid for a few samples."""
        B, T, C, H, W = images.shape
        vis_images = []
        for i in range(min(B, self.num_samples)):
            frames = []
            for t in range(T):
                img_pil = transforms.ToPILImage()(images[i, t])
                fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
                ax.imshow(img_pil)
                ax.axis("off")

                x, y = points[i, t].tolist()
                color = "white"
                v = int(vis_state[i, t].item()) if vis_state is not None else 2
                if v == 2:
                    color = "lime"
                elif v == 1:
                    color = "yellow"
                elif v == 0:
                    color = "red"

                # Draw point
                ax.plot([x], [y], marker="o", color=color, markersize=6, alpha=0.9)

                # Draw speed arrow if provided
                if speed_px is not None:
                    vx, vy = speed_px[i, t].tolist()
                    ax.arrow(
                        x,
                        y,
                        vx,
                        vy,
                        head_width=6,
                        head_length=8,
                        fc="cyan",
                        ec="cyan",
                        length_includes_head=True,
                        alpha=0.8,
                    )

                fig.suptitle(title, fontsize=8)
                fig.tight_layout(pad=0)
                with io.BytesIO() as buf:
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    im = Image.open(buf).convert("RGB")
                    frames.append(transforms.ToTensor()(im))
                plt.close(fig)

            vis_images.append(make_grid(frames, nrow=T))

        return make_grid(vis_images, nrow=1)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not getattr(trainer, "logger", None) or self._buffer is None:
            return
        writer = trainer.logger.experiment

        images: torch.Tensor = self._buffer["images"]
        is_sequence = images.dim() == 5

        # Log trajectory visualization for sequences
        if is_sequence:
            B, T, _, H, W = images.shape

            # Strides from config (ascending resolutions)
            strides: List[int] = []
            if hasattr(pl_module, "cfg") and hasattr(pl_module.cfg.model, "deep_supervision_strides"):
                strides = list(pl_module.cfg.model.deep_supervision_strides)

            # Pred heatmaps -> [B,T,1,Hs,Ws]
            pred_hmaps_raw: List[torch.Tensor] = self._buffer["pred_heatmaps"]
            pred_hmaps_bt = [self._hm_to_bt(hm, B, T) for hm in pred_hmaps_raw]
            pred_hm_hi, stride_hi, Hs, Ws = self._pick_highest_res(pred_hmaps_bt, strides)
            pred_xy_hm = soft_argmax_2d(pred_hm_hi)
            pred_xy_img = pred_xy_hm * float(stride_hi)

            # Target heatmaps -> derive GT coords from highest-res target HM (fallback to true coords if provided later)
            tgt_hmaps_bt: List[torch.Tensor] = [hm for hm in self._buffer["target_heatmaps"]]
            tgt_hm_hi, stride_tgt, _, _ = self._pick_highest_res(tgt_hmaps_bt, strides)
            gt_xy_hm = soft_argmax_2d(tgt_hm_hi)
            gt_xy_img = gt_xy_hm * float(stride_tgt)

            # Valid mask and vis_state
            valid = self._buffer.get("valid_mask")
            vis_state = self._buffer.get("vis_state")
            if vis_state is None and valid is not None:
                vis_state = valid.to(torch.long) * 2  # approx: True->2, False->0

            # Speed (unnormalize by image size)
            speed_norm = self._buffer.get("speed_gt_norm")  # [B,T,2] normalized by [W,H]
            speed_px = None
            if speed_norm is not None:
                scale = torch.tensor([W, H], dtype=torch.float32).view(1, 1, 2)
                speed_px = speed_norm * scale

            # 1) Trajectory on center frame
            traj_grid = self._draw_trajectory_on_grid(
                images, pred_xy_img, gt_xy_img, valid if valid is not None else torch.ones(B, T, dtype=torch.bool)
            )
            writer.add_image("val/trajectory", traj_grid, global_step=pl_module.current_epoch)

            # 2) Per-frame GT points + speed (similar to tools/visualize_dataset.py)
            gt_grid = self._draw_sequence_points_grid(images, gt_xy_img, speed_px, vis_state, title="GT + Speed")
            writer.add_image("val/sequence_gt", gt_grid, global_step=pl_module.current_epoch)

            # 3) Per-frame Predicted points
            pred_grid = self._draw_sequence_points_grid(images, pred_xy_img, None, vis_state, title="Predicted")
            writer.add_image("val/sequence_pred", pred_grid, global_step=pl_module.current_epoch)

            # 4) Heatmaps per scale (optional: both target and prediction)
            if self.draw_multiscale:
                for si, (phm, thm) in enumerate(zip(pred_hmaps_bt, tgt_hmaps_bt)):
                    # Normalize and tile across T for a few samples
                    rows_pred = []
                    rows_tgt = []
                    for i in range(min(B, self.num_samples)):
                        # [T,1,Hs,Ws] -> make_grid over T
                        pred_row = make_grid(phm[i], nrow=T, normalize=True)
                        tgt_row = make_grid(thm[i], nrow=T, normalize=True)
                        rows_pred.append(pred_row)
                        rows_tgt.append(tgt_row)
                    pred_grid_hm = make_grid(rows_pred, nrow=1)
                    tgt_grid_hm = make_grid(rows_tgt, nrow=1)
                    s_val = strides[si] if strides else 1
                    writer.add_image(
                        f"val/heatmaps/stride_{s_val}/pred", pred_grid_hm, global_step=pl_module.current_epoch
                    )
                    writer.add_image(
                        f"val/heatmaps/stride_{s_val}/target", tgt_grid_hm, global_step=pl_module.current_epoch
                    )

        self._buffer = None
