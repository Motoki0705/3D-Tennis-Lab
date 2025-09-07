from __future__ import annotations


class Adversary:
    def d_step(self, module, trajectories_real, trajectories_fake):
        return {}

    def g_loss(self, module, trajectories_fake):
        return 0.0

    def disc_score(self, trajectories):
        return None
