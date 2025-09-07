from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn

from development.ball_tracking.ball_heatmap.losses.gan import gradient_penalty
from .base import Adversary


class SimpleCritic(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class WGAN_GP_Adversary(Adversary):
    def __init__(self, cfg):
        self.cfg = cfg
        self.lambda_gp = float(cfg.gan.lambda_gp)
        self.lambda_adv = float(cfg.gan.lambda_adv)
        # Trajectory feature dim: (x,y,vis,vel,acc) => 2 + 1 + 2 + 2 = 7 per timestep
        self.input_dim = 7 * int(cfg.data.T)
        self.critic = SimpleCritic(self.input_dim)
        self.opt_d = torch.optim.Adam(self.critic.parameters(), lr=2e-4, betas=(0.5, 0.9))

    def _flatten(self, traj: torch.Tensor) -> torch.Tensor:
        return traj.view(traj.size(0), -1)

    def d_step(
        self, module, trajectories_real: torch.Tensor, trajectories_fake: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        self.critic.train()
        real = self._flatten(trajectories_real.detach())
        fake = self._flatten(trajectories_fake.detach())
        d_real = self.critic(real).mean()
        d_fake = self.critic(fake).mean()
        gp = gradient_penalty(self.critic, real, fake)
        loss_d = -(d_real - d_fake) + self.lambda_gp * gp
        self.opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        self.opt_d.step()
        return {"loss_d": loss_d.detach(), "gp": gp.detach(), "d_real": d_real.detach(), "d_fake": d_fake.detach()}

    def g_loss(self, module, trajectories_fake: torch.Tensor) -> torch.Tensor:
        self.critic.eval()
        fake = self._flatten(trajectories_fake)
        return -self.critic(fake).mean() * self.lambda_adv

    def disc_score(self, trajectories: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            s = self.critic(self._flatten(trajectories))
            s = (s - s.min()) / (s.max() - s.min() + 1e-6)
            return s  # [B]
