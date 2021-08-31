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
    nn.init.xavier_uniform_(terminal.weight)
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
    terminal[6] = nn.Linear(in_features=terminal[6].in_features, out_features=n_classes)
    terminal.add_module('7', nn.Softmax(dim=1))
    # terminal.add_module('7', nn.Softmax(dim=0))

    for layer in terminal:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            stdv = 1.0 / (layer.in_features ** 0.5)
            layer.bias.data.uniform_(-stdv, stdv)

    for param in model.parameters():
        param.requires_grad = True

    return model



if __name__ == '__main__':
    model = Vgg19()
    print(model)