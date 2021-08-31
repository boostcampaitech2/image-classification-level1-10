import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2.2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, labels):
        ce_loss = F.cross_entropy(outputs, labels, reduction='none') # to keep per-batch-item loss
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss).mean()
        
        return focal_loss

class FocalSmoothingLoss(nn.Module):
    def __init__(self, total_epoch, gamma=3.0):
        super(FocalSmoothingLoss, self).__init__()
        self.gamma = gamma
        self.curr_ep = 0
        self.total_ep = total_epoch

    def forward(self, outputs, labels):
        focal = 2 + (self.gamma - 2) * (self.curr_ep / self.total_ep)
        return torch.sum(torch.abs(torch.pow(outputs-labels, focal))).mean()

    def step_loss(self):
        self.curr_ep += 1