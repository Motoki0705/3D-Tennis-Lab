import torch.nn as nn
import torch.nn.functional as F

from .loss_registry import register_loss


@register_loss("kldiv")
class KLDivLoss(nn.Module):
    def __init__(self, reduction="batchmean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = F.log_softmax(inputs, dim=1)
        loss = F.kl_div(inputs, targets, reduction=self.reduction)
        return loss
