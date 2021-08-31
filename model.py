import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module) :
    def __init__(self, num_classes) :
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size = 7, stride = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x) :
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x) # 이렇게 다시 자기자신에게 삽입하면 결국 inplace = True와 동일하다.

        x = self.avgpool(x)  # 압축시켰다가
        x = x.view(-1, 128)  # 다시 늘려주는 이유가 있나??
        return self.fc(x)


class CustomResnet_modify_1(nn.Module) :
    '''
    avgpool 이전까지만 가져와서 avgpool을 한 후 size를 조금 더 작게 출력한다.
    '''
    def __init__(self, num_classes) :
        super().__init__()

        self.resnet = torchvision.models.resnet18(pretrained = True)
        self.final_layer = nn.Sequential(
            nn.Linear(128, 18),
        )

    def forward(self, x) :
        x = self.resnet(x)
        x.fc = self.final_layer
        torch.nn.init.xavier_uniform_(x.fc[0].weight)
        stdv = 1.0 / np.sqrt(x.fc[0].in_features)
        x.fc[0].bias.data.uniform_(-stdv, stdv)
        return x

class CustomResnext_modify_2(nn.Module) :
    '''
    avgpool 이전까지만 가져와서 avgpool을 한 후 size를 조금 더 작게 출력한다.
    '''
    def __init__(self, num_classes) :
        super().__init__()

        self.resnext = torchvision.models.resnext50_32x4d(pretrained = True)
        self.final_layer = nn.Sequential(
            nn.Linear(2048, 18),
        )

    def forward(self, x) :
        x = self.resnext(x)
        x.fc = self.final_layer
        torch.nn.init.xavier_uniform_(x.fc[0].weight)
        stdv = 1.0 / np.sqrt(x.fc[0].in_features)
        x.fc[0].bias.data.uniform_(-stdv, stdv)
        return x

    # @staticmethod
    # def weight_modify(model, linaer_layer) :
    #     torch.nn.init.xavier_uniform_(model.linaer_layer.weight)
    #     stdv = 1.0 / np.sqrt(model.linaer_layer.in_features)
    #     model.linear_layer.bias.data.uniform_(-stdv, stdv)
    #     return model

#print(CustomResnext_modify_2(18))