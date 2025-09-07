from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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

        # Loss selection: mse | focal | kl
        # cfg.training.loss can be a dict-like (OmegaConf) with fields:
        #   name: "mse" | "focal" | "kl"
        #   gamma: float (for focal)
        #   tau: float (for kl)
        #   eps: float (numeric stability)
        loss_cfg = getattr(self.cfg.training, "loss", {})

        def _get(cfg_obj, key: str, default):
            try:
                # OmegaConf/DotDict style
                if hasattr(cfg_obj, key):
                    v = getattr(cfg_obj, key)
                    return default if v is None else v
            except Exception:
                pass
            try:
                if isinstance(cfg_obj, dict):
                    return cfg_obj.get(key, default)
            except Exception:
                pass
            return default

        loss_name = str(_get(loss_cfg, "name", "mse")).lower()
        self.loss_kind = loss_name
        self.focal_gamma = float(_get(loss_cfg, "gamma", 2.0))
        self.kl_tau = float(_get(loss_cfg, "tau", 1.0))
        self.loss_eps = float(_get(loss_cfg, "eps", 1e-6))
        if loss_name == "mse":
            self.criterion = nn.MSELoss()
        elif loss_name in ("focal", "kl", "kld", "kl_div"):
            # Use _compute_loss for these kinds
            self.criterion = None  # type: ignore[assignment]
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
        try:
            from ..utils.checkpoint import load_checkpoint_object

            obj = load_checkpoint_object(path)
        except Exception:
            # Fallback to torch.load with permissive behavior
            try:
                obj = torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[call-arg]
            except TypeError:
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
        loss = self._loss(y_hat, y_true)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y_true = _select_io(batch)
        x = self._prepare_x(x)
        y_out = self(x)
        y_hat = self._select_output(y_out)
        loss = self._loss(y_hat, y_true)
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

    # ------------------------
    # Loss impls
    # ------------------------
    def _loss(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.loss_kind == "mse":
            # Standard MSE
            assert self.criterion is not None
            return self.criterion(y_hat, y_true)

        if self.loss_kind == "focal":
            # Binary focal loss on heatmap with soft targets
            eps = self.loss_eps
            gamma = self.focal_gamma
            p = torch.sigmoid(y_hat)
            t = y_true.clamp(0.0, 1.0).to(p.dtype)
            # Compute per-pixel focal loss
            loss_pos = -(t * ((1 - p).clamp_min(eps) ** gamma) * torch.log(p.clamp_min(eps)))
            loss_neg = -((1 - t) * (p.clamp_min(eps) ** gamma) * torch.log((1 - p).clamp_min(eps)))
            loss = loss_pos + loss_neg
            return loss.mean()

        if self.loss_kind in ("kl", "kld", "kl_div"):
            # KL divergence between spatial distributions of pred and target
            eps = max(self.loss_eps, 1e-8)
            tau = max(self.kl_tau, eps)
            B = y_hat.shape[0]
            p = (y_hat.view(B, -1) / tau).to(torch.float32)
            q = (y_true.view(B, -1) / tau).to(torch.float32)
            logp = F.log_softmax(p, dim=-1)
            q = F.softmax(q, dim=-1)
            # batchmean: sum over dims, mean over batch
            return F.kl_div(logp, q, reduction="batchmean")

        raise AssertionError(f"Unknown loss kind set: {self.loss_kind}")


__all__ = ["HRNetFinetuneLitModule"]
