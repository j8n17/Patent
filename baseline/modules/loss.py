import torch
import torch.nn as nn

def get_loss_fn(cfg, pos_weights):
    name = cfg.train.loss.name
    params = cfg.train.loss.params
    # params['pos_weights'] = pos_weights
    loss_fn = LOSS_FUNTIONS[name](**params, pos_weights=pos_weights)
    return loss_fn

class FocalLoss(nn.Module):
    def __init__(self, gamma, pos_weights):
        super().__init__()
        self.gamma = gamma
        self.pos_weights = torch.tensor(pos_weights, dtype=torch.float32)

    def __call__(self, output, label):
        device = label.device  # label 텐서가 위치한 디바이스를 가져옴
        alpha = torch.where(label, self.pos_weights.to(device), torch.tensor(1.0, dtype=torch.float32).to(device))
        p = torch.sigmoid(output)
        pt = torch.where(label, p, 1-p)
        loss = - alpha * (1-pt).pow(self.gamma) * pt.log()
        return loss.mean()

    def __repr__(self):
        return f"FocalLoss(gamma={self.gamma}, pos_weight={self.pos_weights})"

LOSS_FUNTIONS = {
    'FocalLoss': FocalLoss,
}
