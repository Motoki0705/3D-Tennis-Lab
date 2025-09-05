from __future__ import annotations
import torch


def gradient_penalty(disc, real, fake):
    B = real.size(0)
    eps = torch.rand(B, 1, 1, device=real.device)
    inter = eps * real + (1 - eps) * fake
    inter.requires_grad_(True)
    pred = disc(inter)
    grad = torch.autograd.grad(
        outputs=pred,
        inputs=inter,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gp = ((grad.view(B, -1).norm(2, dim=1) - 1) ** 2).mean()
    return gp
