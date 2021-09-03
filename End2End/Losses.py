import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy as CE
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.loss import CrossEntropyLoss

def onehot(y, n_classes=18):
    onehot_label = [0.] * n_classes
    onehot_label[y] = 1.
    return (np.array(onehot_label))
    # return torch.tensor(np.array(onehot_label))


# class FocalLoss(nn.Module):
#     def __init__(self, alpha=.25, gamma=2.):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma

#     def forward(self, outputs, labels):
#         ce_loss = F.cross_entropy(outputs, labels, reduction='none') # to keep per-batch-item loss
#         pt = torch.exp(-ce_loss)
#         # focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss).mean()
#         focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
#         return focal_loss

class LabelSmoothingLoss(nn.Module):
    def __init__(self):
        super(LabelSmoothingLoss, self).__init__()

    def forward(self, output, label_a: int, label_b: int, ratio: float):
        loss = F.cross_entropy(output, label_a) * (1-ratio) + F.cross_entropy(output, label_b) * ratio
        loss = loss.mean()
        return loss

class LabelSmoothingLoss2(nn.Module):
    def __init__(self):
        super(LabelSmoothingLoss, self).__init__()

    def forward(self, output, label_a: int, label_b: int, ratio: float):
        loss = F.cross_entropy(output, label_a) * (1-ratio) + F.cross_entropy(output, label_b) * ratio
        loss = loss.mean()
        return loss



# class LabelSmoothingLossOrigin(nn.Module):
#     def __init__(self, classes, smoothing=0.0, dim=1):
#         super(LabelSmoothingLoss, self).__init__()
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing
#         self.cls = classes
#         self.dim = dim

#     def forward(self, pred, target):
#         pred = pred.softmax(dim=self.dim)
#         # pred = pred.log_softmax(dim=self.dim)
#         with torch.no_grad():
#             true_dist = torch.zeros_like(pred)
#             true_dist.fill_(self.smoothing/(self.cls-1))
#             true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
#             print(true_dist)
#         # return torch.sum(-true_dist * pred, dim=self.dim)
#         return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class SoftmaxProbs(nn.Module):
    def __init__(self, n_classes=18):
        super(SoftmaxProbs, self).__init__()
        self.n_classes = n_classes

    def forward(self, output, labels_soft):
        total_loss = 0.
        transposed = labels_soft.transpose(0,1)

        for cls in range(self.n_classes):
            panel = torch.zeros_like(torch.mean(output, 1), dtype=torch.long)
            panel.fill_(cls)
            total_loss += torch.mean((F.cross_entropy(output, panel, reduction='none') * transposed[cls]))
        
        return total_loss



if __name__ == '__main__':
    # outputs = torch.tensor([[0.7,0.1,0.2], [0.2,0.7,0.1]])
    # labels = torch.tensor([0,1])

    outputs1 = torch.tensor([[0.7, 0.3, 0.], [0.0, 1.0, 0.0]])
    labels_soft = torch.tensor([[0.8, 0.0, 0.2], [0.1, 0.9, 0.0]])
    labels = torch.tensor([1,1])

    ce1 = F.cross_entropy(outputs1, labels, reduction='none')


    labels_soft = labels_soft.transpose(0,1)

    total_loss1 = 0.
    total_loss2 = 0.
    for cls in range(3):
        panel = torch.zeros_like(torch.mean(outputs1,1), dtype=torch.long)
        panel.fill_(cls)
        print(panel)

        total_loss1 += torch.mean((F.cross_entropy(outputs1, panel, reduction='none') * labels_soft[cls]))
    print(total_loss1)
