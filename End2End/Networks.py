import timm
import torch.nn as nn
import torchvision.models as models

n_classes = 18

def Efficientnet_b2():
    model = timm.create_model("efficientnet_b2", pretrained=True)
    terminal = model.classifier
    terminal.out_features=n_classes
    stdv = 1 / terminal.in_features ** 0.5
    terminal.bias.data.uniform_(-stdv, stdv)
    for param in model.parameters():
        param.requires_grad = True
    
    return model


def Vgg16():
    model = models.vgg16(pretrained=True)
    terminal = model.classifier
    terminal[6].out_features = n_classes
    # terminal.add_module(
        # 'terminal',nn.Linear(in_features=terminal[6].out_features, out_features=n_classes))
    
    stdv = 1 / terminal[6].in_features ** 0.5
    terminal[6].bias.data.uniform_(-stdv, stdv)
    for param in model.parameters():
        param.requires_grad = True

    return model


def Vgg19():
    model = models.vgg19(pretrained=True)
    terminal = model.classifier
    terminal.add_module(
        '7', nn.Sequential(
            nn.Linear(in_features=terminal[6].out_features, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=512, out_features=n_classes)
        )
    )
        # '7',nn.Linear(in_features=terminal[6].out_features, out_features=n_classes),)
    
    # terminal[6].out_features = n_classes
    stdv = 1 / terminal[6].in_features ** 0.5
    terminal[6].bias.data.uniform_(-stdv, stdv)
    for param in model.parameters():
        param.requires_grad = True

    return model



if __name__ == '__main__':
    model = Vgg19()
    print(model)