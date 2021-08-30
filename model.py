import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pretrained = torchvision.models.resnext50_32x4d(pretrained=True)
        self.pretrained.fc = nn.Sequential(
            self.pretrained.fc,
            nn.ReLU(inplace = True),
            # nn.Dropout(p = 0.5),
            nn.Linear(1000, 256, bias = True),
            nn.ReLU(inplace = True),
            # nn.Dropout(p = 0.5),
            nn.Linear(256, num_classes, bias = True))

        # initialize weight, bias
        for layer in self.pretrained.fc:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                stdv = 1.0 / (layer.in_features ** 0.5)
                layer.bias.data.uniform_(-stdv,stdv)


    def forward(self, x):
        return self.pretrained(x)

# Custom Model Template
class My3Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet =  torchvision.models.resnext50_32x4d(pretrained=True)

        # for param, weight in self.feature_extraction.named_parameters():
        #     weight.requires_grad = False

        self.mask_classifier = nn.Sequential(
                                nn.ReLU(),
                                nn.Dropout(p = 0.5),
                                nn.Linear(1000, 256, bias = True),
                                nn.ReLU(),
                                nn.Dropout(p = 0.5),
                                nn.Linear(256, 3, bias = True))
            
        self.gender_classifier = nn.Sequential(
                                nn.ReLU(),
                                nn.Dropout(p = 0.5),
                                nn.Linear(1000, 256, bias = True),
                                nn.ReLU(),
                                nn.Dropout(p = 0.5),
                                nn.Linear(256, 2, bias = True))
        
        self.age_classifier = nn.Sequential(
                                nn.ReLU(),
                                nn.Dropout(p = 0.5),
                                nn.Linear(1000, 256, bias = True),
                                nn.ReLU(),
                                nn.Dropout(p = 0.5),
                                nn.Linear(256, 3, bias = True))
        
        
        
        # initialize weight, bias
        for layer in self.mask_classifier:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                stdv = 1.0 / (layer.in_features ** 0.5)
                layer.bias.data.uniform_(-stdv,stdv)

        for layer in self.gender_classifier:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                stdv = 1.0 / (layer.in_features ** 0.5)
                layer.bias.data.uniform_(-stdv,stdv)
                
        for layer in self.age_classifier:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                stdv = 1.0 / (layer.in_features ** 0.5)
                layer.bias.data.uniform_(-stdv,stdv)



    def forward(self, x):
        x = self.resnet(x)
        mask = self.mask_classifier(x)
        gender = self.gender_classifier(x)
        age = self.age_classifier(x)

        return mask, gender, age
