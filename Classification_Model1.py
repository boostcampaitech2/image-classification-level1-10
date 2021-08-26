import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class Model2(nn.Module):
    def __init__(self, num_classes=18):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.linear1 = nn.Sequential(
            nn.Linear(64*30*22, 1000),
            nn.BatchNorm1d(1000)
        )
        self.linear2 = nn.Linear(1000, 18)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class Model1(nn.Module):
    def __init__(self, num_classes=18):
        super(Model1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 16, 3)
        self.linear1 = nn.Linear(16*124*92, 1000)
        self.linear2 = nn.Linear(1000, 18)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class Model(nn.Module):
    def __init__(self, num_classes: int = 18):
        super(Model, self).__init__()
        self.feature_layer = torchvision.models.resnet50(pretrained=True)
        self.linear_layer1 = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU()
        )
        self.linear_layer2 = nn.Linear(500, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_layer(x)
        x = self.linear_layer1(x)
        x = self.linear_layer2(x)
        return x