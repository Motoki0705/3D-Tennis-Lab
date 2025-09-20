"""Lightning module specialisation for the DINO sequential experiment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict

import torch

from development.core.lightning.base_lit_module import BaseLitModule


class SequenceHeatmapLitModule(BaseLitModule):
    """Extends :class:`BaseLitModule` with richer validation outputs for heatmap logging."""

    def __init__(
        self,
        *,
        config,
        model,
        loss_fn,
        metric_fns: Mapping[str, Any] | None = None,
        target_key: str = "heatmaps",
        include_inputs: bool = True,
    ) -> None:
        super().__init__(config=config, model=model, loss_fn=loss_fn, metric_fns=dict(metric_fns or {}))
        self.target_key = target_key
        self.include_inputs = include_inputs

    def validation_step(self, batch, batch_idx):
        inputs, targets = self._split_batch(batch)
        preds = self(inputs)
        loss = self.loss_fn(preds, targets)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)

        for name, fn in self.metric_fns.items():
            val = fn(preds, targets)
            self.log(f"val/{name}", val, prog_bar=True, on_epoch=True)

        outputs = self._build_step_outputs(batch, preds, targets)
        outputs["loss"] = loss.detach()
        return outputs

    def test_step(self, batch, batch_idx):
        inputs, targets = self._split_batch(batch)
        preds = self(inputs)
        loss = self.loss_fn(preds, targets)
        self.log("test/loss", loss, prog_bar=True, on_epoch=True)

        for name, fn in self.metric_fns.items():
            val = fn(preds, targets)
            self.log(f"test/{name}", val, prog_bar=True, on_epoch=True)

        outputs = self._build_step_outputs(batch, preds, targets)
        outputs["loss"] = loss.detach()
        return outputs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_step_outputs(self, batch, preds: torch.Tensor, targets: Any) -> Dict[str, Any]:
        outputs: Dict[str, Any] = {
            "pred_heatmaps": preds.detach(),
        }

        target_tensor = self._extract_target_tensor(targets)
        if target_tensor is not None:
            outputs["target_heatmaps"] = target_tensor.detach()

        if self.include_inputs and isinstance(batch, Mapping):
            inputs = batch.get("inputs")
            if torch.is_tensor(inputs):
                outputs["images"] = inputs.detach()
        return outputs

    def _extract_target_tensor(self, targets: Any) -> torch.Tensor | None:
        if torch.is_tensor(targets):
            return targets
        if isinstance(targets, Mapping):
            value = targets.get(self.target_key)
            if torch.is_tensor(value):
                return value
        return None


__all__ = ["SequenceHeatmapLitModule"]
