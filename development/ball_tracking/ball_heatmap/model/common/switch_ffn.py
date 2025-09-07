from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwitchFFN(nn.Module):
    """
    Transformer FFN with Mixture-of-Experts (top-1 routing).
    Keeps the same call signature as a standard MLP: returns only y, but internally
    accumulates aux losses (router z-loss, load-balance) accessible via .consume_aux().

    Expected input: tokens [B*, L, C]
    """

    def __init__(
        self,
        dim: int,
        hidden_mult: float = 4.0,
        num_experts: int = 4,
        drop: float = 0.0,
        router_z_loss_coef: float = 1e-2,
        load_balance_coef: float = 1e-2,
    ):
        super().__init__()
        self.dim = dim
        self.E = int(num_experts)
        self.hidden = int(round(dim * hidden_mult))
        self.drop = nn.Dropout(drop)
        self.gate = nn.Linear(dim, self.E, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, self.hidden),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(self.hidden, dim),
                nn.Dropout(drop),
            )
            for _ in range(self.E)
        ])
        # aux loss accumulators (summed across calls)
        self._aux_router_z = 0.0
        self._aux_load_bal = 0.0
        self._router_z_coef = float(router_z_loss_coef)
        self._load_bal_coef = float(load_balance_coef)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*, L, C]
        gate_logits = self.gate(x)
        probs = F.softmax(gate_logits, dim=-1)
        top1 = probs.argmax(dim=-1)
        with torch.no_grad():
            mean_probs = probs.mean(dim=(0, 1))
            loads = torch.stack([(top1 == e).float().mean() for e in range(self.E)])
        self._aux_router_z += gate_logits.pow(2).mean() * self._router_z_coef
        self._aux_load_bal += (mean_probs * loads).sum() * self.E * self._load_bal_coef
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            mask = top1 == e
            if mask.any():
                y = expert(x[mask])
                out[mask] = y
        return out

    def consume_aux(self) -> Dict[str, torch.Tensor]:
        aux = {
            "moe_router_z": torch.as_tensor(self._aux_router_z, device=self.gate.weight.device, dtype=torch.float32),
            "moe_load_balance": torch.as_tensor(
                self._aux_load_bal, device=self.gate.weight.device, dtype=torch.float32
            ),
        }
        self._aux_router_z = 0.0
        self._aux_load_bal = 0.0
        return aux
