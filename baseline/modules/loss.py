import torch
import torch.nn as nn

def get_loss_fn(cfg):
    name = cfg.train.loss.name
    params = cfg.train.loss.params
    loss_fn = LOSS_FUNTIONS[name](**params)
    return loss_fn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, pos_weight=1):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def __call__(self, output, label):
        alpha = torch.where(label, self.pos_weight, 1)
        p = torch.sigmoid(output)
        pt = torch.where(label, p, 1-p)
        loss = - alpha * (1-pt).pow(self.gamma) * pt.log()
        return loss.mean()

    def __repr__(self):
        return f"FocalLoss(gamma={self.gamma}, pos_weight={self.pos_weight})"

LOSS_FUNTIONS = {
    'FocalLoss': FocalLoss,
}
