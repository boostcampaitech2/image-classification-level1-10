import numpy as np

import torch
import torchvision
import torch.nn as nn

class CustomModel(nn.Module) :
    def __init__(self, num_classes) :
        super().__init__()

        self.resnext = torchvision.models.resnext50_32x4d(pretrained = True)
        self.final_layer = nn.Sequential(
            nn.Linear(2048, 1024, bias = True),
            nn.ReLU(inplace = True),
            nn.Dropout(p = 0.3),
            nn.Linear(1024, 256, bias = True),
            nn.ReLU(inplace = True),
            nn.Dropout(p = 0.3),
            nn.Linear(256, num_classes, bias = True)
        )
        self.resnext.fc = self.final_layer
        torch.nn.init.xavier_uniform_(self.resnext.fc[0].weight)
        stdv = 1.0 / np.sqrt(self.resnext.fc[0].in_features)
        self.resnext.fc[0].bias.data.uniform_(-stdv, stdv)
        
    def forward(self, x) :
        x = self.resnext(x)
        return x
