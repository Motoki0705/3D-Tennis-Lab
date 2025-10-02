from __future__ import annotations

from typing import Any, Mapping

import torch

from development.core.lightning.base_lit_module import BaseLitModule


class CourtPoseLitModule(BaseLitModule):
    """LightningModule wrapper that returns tensors for the pluggable heatmap logger."""

    def __init__(
        self,
        cfg,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        metric_fns: Mapping[str, Any] | None = None,
        *,
        target_key: str = "heatmaps",
        include_inputs: bool = True,
    ) -> None:
        super().__init__(cfg, model=model, loss_fn=loss_fn, metric_fns=metric_fns or {})
        self.target_key = target_key
        self.include_inputs = include_inputs

    # ------------------------------------------------------------------
    # Training hook with target resolution for dict batches
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx: int):
        images, targets = self._split_batch(batch)
        target = self._resolve_target(targets)
        preds = self(images)

        loss = self.loss_fn(preds, target)
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self._infer_batch_size(images),
        )

        for name, fn in self.metric_fns.items():
            metric_val = fn(preds, target)
            self.log(
                f"train/{name}",
                metric_val,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=self._infer_batch_size(images),
            )

        return loss

    # ------------------------------------------------------------------
    # Evaluation hooks with logging payloads for callbacks
    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx: int):
        return self._eval_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx: int):
        return self._eval_step(batch, batch_idx, stage="test")

    def _eval_step(self, batch, batch_idx: int, *, stage: str):
        images, targets = self._split_batch(batch)
        preds = self(images)
        target = self._resolve_target(targets)

        loss = self.loss_fn(preds, target)
        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            batch_size=self._infer_batch_size(images),
        )

        for name, fn in self.metric_fns.items():
            metric_val = fn(preds, target)
            self.log(
                f"{stage}/{name}",
                metric_val,
                prog_bar=True,
                on_epoch=True,
                batch_size=self._infer_batch_size(images),
            )

        payload = {
            "loss": loss.detach(),
            "pred_heatmaps": preds.detach(),
            "target_heatmaps": target.detach(),
        }
        if self.include_inputs and torch.is_tensor(images):
            payload["images"] = images.detach()
        return payload

    def _resolve_target(self, targets: Any) -> torch.Tensor:
        if torch.is_tensor(targets):
            return targets
        if isinstance(targets, Mapping):
            if self.target_key not in targets:
                raise KeyError(f"Target mapping missing key '{self.target_key}'. Available: {list(targets.keys())}")
            value = targets[self.target_key]
            if torch.is_tensor(value):
                return value
            raise TypeError(f"Expected tensor for targets['{self.target_key}'], got {type(value)}")
        raise TypeError(f"Unsupported targets type: {type(targets)}")

    @staticmethod
    def _infer_batch_size(images: Any) -> int | None:
        if torch.is_tensor(images) and images.dim() >= 1:
            return int(images.size(0))
        return None


__all__ = ["CourtPoseLitModule"]
