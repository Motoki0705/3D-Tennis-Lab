"""Lightning wrapper for DINOv3 + DETRPose."""

from __future__ import annotations
from typing import Any, Dict, Mapping, Optional
import torch

from development.core.lightning.base_lit_module import BaseLitModule


class DinoDetrPoseLitModule(BaseLitModule):
    def __init__(
        self,
        *,
        cfg: Any,
        model,
        loss_fn,
        metric_fns: Optional[Mapping[str, Any]] = None,
        postprocessors: Mapping[str, Any] | None = None,
        log_max_predictions: int = 10,
        log_image_mean=(0.485, 0.456, 0.406),
        log_image_std=(0.229, 0.224, 0.225),
    ) -> None:
        super().__init__(config=cfg, model=model, loss_fn=loss_fn, metric_fns=dict(metric_fns or {}))
        self.postprocessors = dict(postprocessors or {})
        self.max_preds = int(log_max_predictions)
        self.log_mean = torch.tensor(log_image_mean, dtype=torch.float32).view(1, -1, 1, 1)
        self.log_std = torch.tensor(log_image_std, dtype=torch.float32).view(1, -1, 1, 1)
        pose_processor = self.postprocessors.get("pose")
        self.num_body_points = getattr(pose_processor, "num_body_points", 17)

    # ---------------------------------------------------------------
    def forward(self, images: torch.Tensor, targets: Optional[list[Dict[str, torch.Tensor]]] = None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx: int):
        images, targets = batch
        outputs = self.model(images, targets)
        loss_dict = self.loss_fn(outputs, targets)
        total = torch.zeros((), device=images.device, dtype=torch.float32)
        for name, value in loss_dict.items():
            if torch.is_tensor(value):
                self.log(f"train/{name}", value, on_step=True, on_epoch=True, prog_bar=False, batch_size=len(images))
                total = total + value
        self.log("train/loss", total, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(images))
        return total

    def validation_step(self, batch, batch_idx: int):
        images, targets = batch
        outputs = self.model(images, targets)

        # ---- PATCH: ensure aux_outputs exists on eval so Criterion won't assert
        outputs_for_loss = self._ensure_aux_for_eval(outputs)

        loss_dict = self.loss_fn(outputs_for_loss, targets)
        total = torch.zeros((), device=images.device, dtype=torch.float32)
        for name, value in loss_dict.items():
            if torch.is_tensor(value):
                self.log(f"val/{name}", value, on_step=False, on_epoch=True, prog_bar=False, batch_size=len(images))
                total = total + value
        self.log("val/loss", total, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(images))

        for metric_name, metric_fn in self.metric_fns.items():
            metric_val = metric_fn(outputs, targets)
            if torch.is_tensor(metric_val):
                self.log(f"val/{metric_name}", metric_val, on_step=False, on_epoch=True, batch_size=len(images))

        render_payload = self._build_render_payload(images, targets, outputs)
        render_payload["val_loss"] = total.detach()
        return render_payload

    # ---------------------------------------------------------------
    def _ensure_aux_for_eval(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        On eval, Criterion expects 'aux_outputs', 'aux_pre_outputs', and 'aux_interm_outputs'.
        Ensure they exist. To avoid matcher crashes, make aux_pre_outputs mirror the final preds
        (not zero-length), while keeping aux_outputs/aux_interm_outputs empty lists.
        """
        if self.training or not isinstance(outputs, dict):
            return outputs

        # preserve insertion order so Criterion's device inference picks a Tensor first
        patched = dict(outputs)

        pl = patched.get("pred_logits")
        pk = patched.get("pred_keypoints")
        if pl is None or pk is None:
            return outputs  # let it fail loudly; model didn't return required keys

        # 1) always present, but empty on eval (no aux decoder layers)
        patched.setdefault("aux_outputs", [])
        patched.setdefault("aux_interm_outputs", [])

        # 2) aux_pre_outputs must have a **valid Q** to satisfy the matcher
        #    -> mirror the final head tensors (no detach needed)
        if "aux_pre_outputs" not in patched:
            patched["aux_pre_outputs"] = {
                "pred_logits": pl,  # shape [B, Q, C]
                "pred_keypoints": pk,  # shape [B, Q, 2*K]
            }

        return patched

    # ---------------------------------------------------------------
    def _build_render_payload(self, images, targets, outputs):
        pose_processor = self.postprocessors.get("pose")
        if pose_processor is None:
            return {}

        device = images.device
        target_sizes = torch.stack([t["size"] for t in targets], dim=0).to(device=device, dtype=torch.float32)
        results = pose_processor(outputs, target_sizes)

        images_denorm = (images.detach().cpu() * self.log_std) + self.log_mean
        images_denorm = images_denorm.clamp(0, 1)

        batch_size = images.shape[0]
        kp_dim = (self.num_body_points, 3)
        max_preds = max(1, self.max_preds)

        pred_keypoints = torch.zeros((batch_size, max_preds, *kp_dim), dtype=torch.float32)
        pred_scores = torch.zeros((batch_size, max_preds), dtype=torch.float32)
        pred_labels = torch.zeros((batch_size, max_preds), dtype=torch.int64)

        gt_keypoints = torch.zeros((batch_size, max_preds, *kp_dim), dtype=torch.float32)
        gt_vis = torch.zeros((batch_size, max_preds), dtype=torch.float32)

        for idx, result in enumerate(results):
            size_hw = targets[idx]["size"].to(torch.float32)
            h, w = size_hw[0].item(), size_hw[1].item()

            det_scores = result.get("scores", torch.zeros((0,), device=device))
            det_labels = result.get("labels", torch.zeros((0,), device=device, dtype=torch.int64))
            det_keypoints = result.get("keypoints", torch.zeros((0, self.num_body_points * 3), device=device))

            keep = min(det_keypoints.size(0), max_preds)
            if keep > 0:
                kp = det_keypoints[:keep].view(keep, self.num_body_points, 3).detach().cpu()
                kp[..., 0] /= max(w, 1.0)
                kp[..., 1] /= max(h, 1.0)
                pred_keypoints[idx, :keep] = kp
                pred_scores[idx, :keep] = det_scores[:keep].detach().cpu()
                pred_labels[idx, :keep] = det_labels[:keep].detach().cpu()

            # Ground truth
            gt = targets[idx]["keypoints"]
            if torch.is_tensor(gt) and gt.numel() > 0:
                gt = gt.detach().cpu()
                gt_count = min(gt.size(0), max_preds)
                xy = gt[:gt_count, : self.num_body_points * 2].view(gt_count, self.num_body_points, 2)
                vis = gt[:gt_count, self.num_body_points * 2 :].view(gt_count, self.num_body_points, 1)
                gt_keypoints[idx, :gt_count] = torch.cat([xy, vis], dim=2)
                gt_vis[idx, :gt_count] = (vis.squeeze(-1) > 0).float().mean(dim=1)

        return {
            "images": images_denorm,
            "pred_keypoints": pred_keypoints,
            "pred_scores": pred_scores,
            "pred_labels": pred_labels,
            "target_keypoints": gt_keypoints,
            "target_visibility": gt_vis,
        }


__all__ = ["DinoDetrPoseLitModule"]
