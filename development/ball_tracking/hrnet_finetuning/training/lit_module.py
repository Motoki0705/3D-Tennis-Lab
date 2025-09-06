from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from hydra.utils import to_absolute_path as abspath

from ..model.base_hrnet_3dstem import HRNet3DStem


logger = logging.getLogger(__name__)


def _select_io(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    # Accept (x, y) or dict with keys
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    if isinstance(batch, dict):
        if "inputs" in batch and "targets" in batch:
            return batch["inputs"], batch["targets"]
        if "x" in batch and "y" in batch:
            return batch["x"], batch["y"]
    raise ValueError("Unsupported batch format. Expected (x,y) or dict with inputs/targets.")


class HRNetFinetuneLitModule(pl.LightningModule):
    """
    LightningModule for finetuning HRNet with a 3D stem.

    Features
    - External checkpoint loading (2D→3D stem変換済み) via config or method
    - Cosine LR with warmup
    - Freeze all except 3D stem for first N epochs
    """

    def __init__(self, config: Any):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.cfg = config

        # Build model
        self.model = HRNet3DStem(self.cfg.model)

        # Loss
        loss_name = str(getattr(self.cfg.training, "loss", {}).get("name", "mse")).lower()
        if loss_name == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss: {loss_name}")

        # Optional: load external pretrained checkpoint
        ckpt_path = getattr(self.cfg.training, "pretrained_checkpoint", None)
        if ckpt_path:
            self.load_external_checkpoint(ckpt_path)

        # Epoch-based freezing control
        self.freeze_epochs = int(getattr(self.cfg.training, "freeze_epochs", 0))

    # ------------------------
    # External checkpoint load
    # ------------------------
    def load_external_checkpoint(self, ckpt_path: str, strict: bool = False):
        path = abspath(ckpt_path)
        obj = torch.load(path, map_location="cpu")
        state_dict: Dict[str, torch.Tensor]
        if isinstance(obj, dict):
            for k in ["state_dict", "model_state_dict", "model_state", "model", "weights"]:
                if k in obj and isinstance(obj[k], dict):
                    state_dict = obj[k]
                    break
            else:
                # plain state_dict
                state_dict = obj if any(isinstance(v, torch.Tensor) for v in obj.values()) else None
        else:
            state_dict = None
        if state_dict is None:
            raise ValueError(f"Unsupported checkpoint format: {path}")

        cleaned = {}
        for k, v in state_dict.items():
            nk = k[len("model.") :] if k.startswith("model.") else k
            cleaned[nk] = v
        missing, unexpected = self.model.load_state_dict(cleaned, strict=strict)
        if missing:
            logger.warning(f"Missing keys while loading: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys while loading: {unexpected}")
        logger.info(f"Loaded pretrained weights from: {path}")

    # ------------------------
    # Freezing policy
    # ------------------------
    def _set_freeze_except_3dstem(self, freeze_others: bool):
        # 3D stem modules
        stem_params = set()
        for name, p in self.model.named_parameters():
            if name.startswith("conv1") or name.startswith("bn1") or name.startswith("conv2") or name.startswith("bn2"):
                stem_params.add(p)

        for p in self.model.parameters():
            if p in stem_params:
                p.requires_grad = True
            else:
                p.requires_grad = not freeze_others

    def on_train_epoch_start(self) -> None:
        if self.freeze_epochs > 0:
            freeze_others = self.current_epoch < self.freeze_epochs
            self._set_freeze_except_3dstem(freeze_others)

    # ------------------------
    # Lightning steps
    # ------------------------
    def forward(self, x: torch.Tensor) -> Any:
        return self.model(x)

    def _select_output(self, y_out: Dict[int, torch.Tensor]) -> torch.Tensor:
        scale = self.cfg.model.out_scales[0]
        y = y_out[scale]
        return y

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y_true = _select_io(batch)
        x = self._prepare_x(x)
        y_out = self(x)
        y_hat = self._select_output(y_out)
        loss = self.criterion(y_hat, y_true)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y_true = _select_io(batch)
        x = self._prepare_x(x)
        y_out = self(x)
        y_hat = self._select_output(y_out)
        loss = self.criterion(y_hat, y_true)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # ------------------------
    # Optimizers & LR schedulers
    # ------------------------
    def configure_optimizers(self):
        lr = float(getattr(self.cfg.training, "lr", 1e-3))
        wd = float(getattr(self.cfg.training, "weight_decay", 1e-4))
        betas = tuple(getattr(self.cfg.training, "betas", (0.9, 0.999)))
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=wd, betas=betas)

        max_epochs = int(getattr(self.cfg.training, "max_epochs", 100))
        warmup_epochs = int(getattr(self.cfg.training, "warmup_epochs", 0))
        cosine_eta_min = float(getattr(self.cfg.training, "eta_min", 1e-6))

        scheds = []
        milestones = []
        if warmup_epochs > 0:
            warmup = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
            )
            scheds.append(warmup)
            milestones.append(warmup_epochs)

        T_max = max(1, max_epochs - warmup_epochs)
        cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=cosine_eta_min)
        if scheds:
            sch = optim.lr_scheduler.SequentialLR(optimizer, schedulers=scheds + [cosine], milestones=milestones)
        else:
            sch = cosine

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sch,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/loss",
            },
        }

    # ------------------------
    # Inputs
    # ------------------------
    def _prepare_x(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, C, T, H, W) or (B, 3*T, H, W)
        if x.dim() == 5:
            B, C, T, H, W = x.shape
            assert C == 3, f"Expected 3 channels, got {C}"
            x = x.permute(0, 1, 2, 3, 4).contiguous().view(B, 3 * T, H, W)
            return x
        elif x.dim() == 4:
            return x
        else:
            raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")


__all__ = ["HRNetFinetuneLitModule"]
