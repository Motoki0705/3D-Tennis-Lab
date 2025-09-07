from __future__ import annotations
import torch


def product_of_experts(mu_det, cov_det, mu_kf, cov_kf):
    inv = torch.inverse(cov_det) + torch.inverse(cov_kf)
    cov_fused = torch.inverse(inv)
    mu_fused = cov_fused @ (torch.inverse(cov_det) @ mu_det + torch.inverse(cov_kf) @ mu_kf)
    return mu_fused, cov_fused
